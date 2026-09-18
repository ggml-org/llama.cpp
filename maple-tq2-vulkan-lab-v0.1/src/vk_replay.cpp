// Standalone Vulkan execution/validation harness. Not a llama.cpp backend patch.
// Build with Vulkan SDK. Only system Vulkan APIs and C++17 are required.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs=std::filesystem;
static void ck(VkResult r,const char * what){if(r!=VK_SUCCESS)throw std::runtime_error(std::string(what)+": VkResult="+std::to_string(r));}
static bool identifier(const std::string&s){return !s.empty()&&std::all_of(s.begin(),s.end(),[](unsigned char c){return std::isalnum(c)||c=='_'||c=='-';});}
static fs::path local_file(const fs::path &root,const std::string&s){fs::path p(s);if(p.is_absolute())throw std::runtime_error("absolute fixture path rejected");for(const auto &v:p)if(v=="..")throw std::runtime_error("parent path rejected");return root/p;}
static std::vector<std::uint8_t> read_file(const fs::path&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("cannot open "+p.string());const auto pos=f.tellg();if(pos<0)throw std::runtime_error("bad file length");std::vector<std::uint8_t>b(static_cast<std::size_t>(pos));f.seekg(0);if(!b.empty()&&!f.read(reinterpret_cast<char*>(b.data()),static_cast<std::streamsize>(b.size())))throw std::runtime_error("read failed "+p.string());return b;}
struct BufferSpec {std::string name,input;std::uint64_t bytes=0;};
struct Op {enum Kind{Fill,Dispatch}kind;std::string target;std::uint64_t offset=0,bytes=0;std::uint32_t value=0;std::array<std::uint32_t,3>grid{};std::vector<std::uint32_t>push;std::vector<std::string>bindings;VkDescriptorSet set=VK_NULL_HANDLE;};
struct Plan {bool dot=false;fs::path root;std::vector<BufferSpec>buffers;std::vector<Op>ops;std::vector<std::pair<std::string,std::string>>dumps;};
static Plan load_plan(const fs::path&path){
 Plan p;p.root=fs::absolute(path).parent_path();std::ifstream f(path);if(!f)throw std::runtime_error("cannot open plan");
 std::string magic;int version=0;f>>magic>>version;if(magic!="MAPLE_TQ2_PLAN"||version!=1)throw std::runtime_error("unsupported plan header");
 std::string line;std::getline(f,line);std::size_t number=1;std::map<std::string,std::uint64_t>sizes;
 while(std::getline(f,line)){++number;if(line.empty()||line[0]=='#')continue;std::istringstream ss(line);std::string cmd;ss>>cmd;
  if(cmd=="require_dot"){int v=-1;ss>>v;if(v!=0&&v!=1)throw std::runtime_error("bad require_dot");p.dot=v!=0;}
  else if(cmd=="buffer"){BufferSpec b;ss>>b.name>>b.bytes>>std::quoted(b.input);if(!identifier(b.name)||sizes.count(b.name)||!b.bytes||b.bytes%4||b.bytes>=0x100000000ull)throw std::runtime_error("invalid buffer");sizes[b.name]=b.bytes;p.buffers.push_back(b);}
  else if(cmd=="fill"){Op o{};o.kind=Op::Fill;ss>>o.target>>o.offset>>o.bytes>>o.value;if(!sizes.count(o.target)||!o.bytes||o.offset%4||o.bytes%4||o.offset>sizes[o.target]||o.bytes>sizes[o.target]-o.offset)throw std::runtime_error("invalid fill range");p.ops.push_back(o);}
  else if(cmd=="dispatch"){Op o{};o.kind=Op::Dispatch;unsigned count=0;ss>>o.target>>o.grid[0]>>o.grid[1]>>o.grid[2]>>count;if(!identifier(o.target)||count>32)throw std::runtime_error("bad kernel/push size");o.push.resize(count);for(auto &v:o.push)ss>>v;ss>>count;if(count>9)throw std::runtime_error("too many bindings");o.bindings.resize(count);for(auto &v:o.bindings){ss>>v;if(!sizes.count(v))throw std::runtime_error("unknown buffer "+v);}for(auto g:o.grid)if(!g)throw std::runtime_error("zero dispatch");p.ops.push_back(o);}
  else if(cmd=="dump"){std::string b,out;ss>>b>>std::quoted(out);if(!sizes.count(b))throw std::runtime_error("unknown dump buffer");local_file(p.root,out);p.dumps.emplace_back(b,out);}
  else throw std::runtime_error("unknown plan command "+cmd);
  if(ss.fail())throw std::runtime_error("malformed plan line "+std::to_string(number));
  std::string trailing;if(ss>>trailing)throw std::runtime_error("extra text on plan line "+std::to_string(number));
 }
 if(!sizes.count("dummy")||p.ops.empty())throw std::runtime_error("empty/incomplete plan");
 for(const auto&o:p.ops)if(o.kind==Op::Dispatch&&o.target.rfind("project_q1_",0)==0&&!p.dot)throw std::runtime_error("INT8 shader requires require_dot 1");
 return p;
}
struct Buffer{VkBuffer handle=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;VkDeviceSize bytes=0;bool coherent=false;};
struct Context{
 VkInstance instance=VK_NULL_HANDLE;VkPhysicalDevice physical=VK_NULL_HANDLE;VkDevice device=VK_NULL_HANDLE;VkQueue queue=VK_NULL_HANDLE;
 VkPhysicalDeviceProperties prop{};VkPhysicalDeviceMemoryProperties mem{};std::uint32_t family=0,timestamp_bits=0;
 VkCommandPool command_pool=VK_NULL_HANDLE;VkDescriptorPool descriptor_pool=VK_NULL_HANDLE;VkDescriptorSetLayout set_layout=VK_NULL_HANDLE;VkPipelineLayout pipeline_layout=VK_NULL_HANDLE;
 VkQueryPool query_pool=VK_NULL_HANDLE;std::map<std::string,VkPipeline>pipelines;std::map<std::string,Buffer>buffers;
 std::uint64_t max_bytes=0,allocated_payload=0;
 ~Context(){if(device){vkDeviceWaitIdle(device);for(auto &v:pipelines)vkDestroyPipeline(device,v.second,nullptr);for(auto&v:buffers)destroy(v.second);if(query_pool)vkDestroyQueryPool(device,query_pool,nullptr);if(pipeline_layout)vkDestroyPipelineLayout(device,pipeline_layout,nullptr);if(set_layout)vkDestroyDescriptorSetLayout(device,set_layout,nullptr);if(descriptor_pool)vkDestroyDescriptorPool(device,descriptor_pool,nullptr);if(command_pool)vkDestroyCommandPool(device,command_pool,nullptr);vkDestroyDevice(device,nullptr);}if(instance)vkDestroyInstance(instance,nullptr);}
 std::uint32_t memory_type(std::uint32_t mask,VkMemoryPropertyFlags required,VkMemoryPropertyFlags preferred){for(int pass=0;pass<2;++pass)for(std::uint32_t i=0;i<mem.memoryTypeCount;++i){auto flags=mem.memoryTypes[i].propertyFlags;if((mask&(1u<<i))&&(flags&required)==required&&(pass||((flags&preferred)==preferred)))return i;}throw std::runtime_error("no suitable memory type");}
 Buffer allocate(VkDeviceSize bytes,bool staging){
  Buffer b;b.bytes=bytes;VkBufferCreateInfo bi{};bi.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;bi.size=bytes;bi.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|(staging?0:VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;ck(vkCreateBuffer(device,&bi,nullptr,&b.handle),"vkCreateBuffer");
  try{VkMemoryRequirements req{};vkGetBufferMemoryRequirements(device,b.handle,&req);auto type=memory_type(req.memoryTypeBits,staging?VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,staging?VK_MEMORY_PROPERTY_HOST_COHERENT_BIT:VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);b.coherent=(mem.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)!=0;
   VkMemoryAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;ai.allocationSize=req.size;ai.memoryTypeIndex=type;ck(vkAllocateMemory(device,&ai,nullptr,&b.memory),"vkAllocateMemory");ck(vkBindBufferMemory(device,b.handle,b.memory,0),"vkBindBufferMemory");
  }catch(...){destroy(b);throw;}return b;
 }
 void destroy(Buffer&b){if(b.handle)vkDestroyBuffer(device,b.handle,nullptr);if(b.memory)vkFreeMemory(device,b.memory,nullptr);b={};}
 VkCommandBuffer begin(){VkCommandBufferAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;ai.commandPool=command_pool;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=1;VkCommandBuffer cmd{};ck(vkAllocateCommandBuffers(device,&ai,&cmd),"allocate commands");VkCommandBufferBeginInfo bi{};bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;ck(vkBeginCommandBuffer(cmd,&bi),"begin commands");return cmd;}
 void submit(VkCommandBuffer cmd){ck(vkEndCommandBuffer(cmd),"end commands");VkSubmitInfo si{};si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;si.commandBufferCount=1;si.pCommandBuffers=&cmd;ck(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE),"queue submit");ck(vkQueueWaitIdle(queue),"queue wait");vkFreeCommandBuffers(device,command_pool,1,&cmd);}
 static void barrier(VkCommandBuffer cmd){VkMemoryBarrier m{};m.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;m.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;m.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;auto stage=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT;vkCmdPipelineBarrier(cmd,stage,stage,0,1,&m,0,nullptr,0,nullptr);}
 void init(unsigned selected,bool need_dot,std::uint64_t memory_limit,unsigned descriptor_sets,unsigned query_count,bool validation){
  max_bytes=memory_limit;
  std::uint32_t loader_version=VK_API_VERSION_1_0;auto enum_version=reinterpret_cast<PFN_vkEnumerateInstanceVersion>(vkGetInstanceProcAddr(VK_NULL_HANDLE,"vkEnumerateInstanceVersion"));if(enum_version)ck(enum_version(&loader_version),"instance version");if(loader_version<VK_API_VERSION_1_2)throw std::runtime_error("Vulkan 1.2 loader required");
  const auto api=std::min(loader_version,std::uint32_t(VK_API_VERSION_1_3));VkApplicationInfo app{};app.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;app.pApplicationName="maple-tq2-lab";app.apiVersion=api;
  const char*validation_name="VK_LAYER_KHRONOS_validation";VkInstanceCreateInfo ii{};ii.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ii.pApplicationInfo=&app;
  if(validation){std::uint32_t nl=0;ck(vkEnumerateInstanceLayerProperties(&nl,nullptr),"enumerate layers");std::vector<VkLayerProperties>layers(nl);ck(vkEnumerateInstanceLayerProperties(&nl,layers.data()),"enumerate layers");bool has=false;for(auto &l:layers)if(std::strcmp(l.layerName,validation_name)==0)has=true;if(!has)throw std::runtime_error("requested validation layer is missing");ii.enabledLayerCount=1;ii.ppEnabledLayerNames=&validation_name;}
  ck(vkCreateInstance(&ii,nullptr,&instance),"create instance");std::uint32_t n=0;ck(vkEnumeratePhysicalDevices(instance,&n,nullptr),"enumerate devices");std::vector<VkPhysicalDevice>d(n);ck(vkEnumeratePhysicalDevices(instance,&n,d.data()),"enumerate devices");if(selected>=n)throw std::runtime_error("device index out of range; use --list");physical=d[selected];vkGetPhysicalDeviceProperties(physical,&prop);vkGetPhysicalDeviceMemoryProperties(physical,&mem);
  if(prop.apiVersion<VK_API_VERSION_1_2)throw std::runtime_error("Vulkan 1.2 device required");
  if(prop.limits.maxPerStageDescriptorStorageBuffers<9||prop.limits.maxDescriptorSetStorageBuffers<9||prop.limits.maxComputeSharedMemorySize<24576||prop.limits.maxComputeWorkGroupInvocations<256||prop.limits.maxComputeWorkGroupSize[0]<256)throw std::runtime_error("device does not meet prototype storage/shared-memory/workgroup limits");
  std::uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,nullptr);std::vector<VkQueueFamilyProperties>qp(nq);vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,qp.data());bool found=false;for(unsigned q=0;q<nq;++q)if(qp[q].queueFlags&VK_QUEUE_COMPUTE_BIT){family=q;timestamp_bits=qp[q].timestampValidBits;found=true;break;}if(!found)throw std::runtime_error("no compute queue");
  std::vector<const char*>extensions;VkPhysicalDeviceShaderIntegerDotProductFeatures df{};df.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;VkPhysicalDeviceFeatures2 f2{};f2.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;f2.pNext=&df;vkGetPhysicalDeviceFeatures2(physical,&f2);
  if(need_dot&&!df.shaderIntegerDotProduct)throw std::runtime_error("shaderIntegerDotProduct unavailable: do not silently substitute float");
  if(need_dot&&std::min(api,prop.apiVersion)<VK_API_VERSION_1_3){std::uint32_t ne=0;ck(vkEnumerateDeviceExtensionProperties(physical,nullptr,&ne,nullptr),"device extensions");std::vector<VkExtensionProperties>ep(ne);ck(vkEnumerateDeviceExtensionProperties(physical,nullptr,&ne,ep.data()),"device extensions");bool has=false;for(auto &e:ep)if(std::strcmp(e.extensionName,VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)==0)has=true;if(!has)throw std::runtime_error("VK_KHR_shader_integer_dot_product required");extensions.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);}
  const float priority=1;VkDeviceQueueCreateInfo qi{};qi.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;VkDeviceCreateInfo di{};di.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;di.enabledExtensionCount=static_cast<std::uint32_t>(extensions.size());di.ppEnabledExtensionNames=extensions.data();df.shaderIntegerDotProduct=need_dot?VK_TRUE:VK_FALSE;di.pNext=need_dot?&df:nullptr;ck(vkCreateDevice(physical,&di,nullptr,&device),"create device");vkGetDeviceQueue(device,family,0,&queue);
  VkCommandPoolCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;ci.queueFamilyIndex=family;ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;ck(vkCreateCommandPool(device,&ci,nullptr,&command_pool),"command pool");
  std::array<VkDescriptorSetLayoutBinding,9>lb{};for(unsigned j=0;j<9;++j){lb[j].binding=j;lb[j].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;lb[j].descriptorCount=1;lb[j].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
  VkDescriptorSetLayoutCreateInfo li{};li.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;li.bindingCount=9;li.pBindings=lb.data();ck(vkCreateDescriptorSetLayout(device,&li,nullptr,&set_layout),"set layout");VkPushConstantRange range{};range.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;range.size=128;VkPipelineLayoutCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pi.setLayoutCount=1;pi.pSetLayouts=&set_layout;pi.pushConstantRangeCount=1;pi.pPushConstantRanges=&range;ck(vkCreatePipelineLayout(device,&pi,nullptr,&pipeline_layout),"pipeline layout");
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,descriptor_sets*9};VkDescriptorPoolCreateInfo dp{};dp.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;dp.maxSets=descriptor_sets;dp.poolSizeCount=1;dp.pPoolSizes=&ps;ck(vkCreateDescriptorPool(device,&dp,nullptr,&descriptor_pool),"descriptor pool");
  if(timestamp_bits&&query_count){VkQueryPoolCreateInfo q{};q.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;q.queryType=VK_QUERY_TYPE_TIMESTAMP;q.queryCount=query_count;ck(vkCreateQueryPool(device,&q,nullptr,&query_pool),"query pool");}
  VkPhysicalDeviceShaderIntegerDotProductProperties dotprops{};dotprops.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;VkPhysicalDeviceProperties2 props2{};props2.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;props2.pNext=&dotprops;vkGetPhysicalDeviceProperties2(physical,&props2);
  std::cout<<"Packed signed INT8 dot accelerated: "<<dotprops.integerDotProduct4x8BitPackedSignedAccelerated<<"\n";
  std::cout<<"Device: "<<prop.deviceName<<", driver="<<prop.driverVersion<<", integer_dot_enabled="<<need_dot<<", shared_memory="<<prop.limits.maxComputeSharedMemorySize<<"\n";
 }
 void load_buffers(const Plan&p){
  for(const auto&s:p.buffers){if(s.bytes>max_bytes-allocated_payload)throw std::runtime_error("device payload exceeds --memory-limit-mib");if(s.bytes>prop.limits.maxStorageBufferRange)throw std::runtime_error("buffer exceeds maxStorageBufferRange; chunk the workload");allocated_payload+=s.bytes;buffers.emplace(s.name,allocate(s.bytes,false));}
  for(const auto&s:p.buffers){auto &b=buffers.at(s.name);if(s.input=="-"){auto cmd=begin();vkCmdFillBuffer(cmd,b.handle,0,b.bytes,0);barrier(cmd);submit(cmd);continue;}
   auto data=read_file(local_file(p.root,s.input));if(data.size()!=s.bytes)throw std::runtime_error("fixture buffer length mismatch: "+s.name);auto staging=allocate(b.bytes,true);
   try{void *ptr=nullptr;ck(vkMapMemory(device,staging.memory,0,VK_WHOLE_SIZE,0,&ptr),"map upload");std::memcpy(ptr,data.data(),data.size());if(!staging.coherent){VkMappedMemoryRange r{};r.sType=VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;r.memory=staging.memory;r.size=VK_WHOLE_SIZE;ck(vkFlushMappedMemoryRanges(device,1,&r),"flush upload");}vkUnmapMemory(device,staging.memory);auto cmd=begin();VkBufferCopy c{0,0,b.bytes};vkCmdCopyBuffer(cmd,staging.handle,b.handle,1,&c);barrier(cmd);submit(cmd);}catch(...){destroy(staging);throw;}destroy(staging);
  }
 }
 void prepare(Plan&p,const fs::path&shaders){
  for(auto&o:p.ops)if(o.kind==Op::Dispatch){for(unsigned axis=0;axis<3;++axis)if(o.grid[axis]>prop.limits.maxComputeWorkGroupCount[axis])throw std::runtime_error("dispatch exceeds device limits");
   if(!pipelines.count(o.target)){auto raw=read_file(shaders/(o.target+".spv"));if(raw.empty()||raw.size()%4)throw std::runtime_error("bad SPIR-V byte size");std::vector<std::uint32_t>code(raw.size()/4);std::memcpy(code.data(),raw.data(),raw.size());if(code[0]!=0x07230203u)throw std::runtime_error("bad SPIR-V magic");VkShaderModuleCreateInfo sm{};sm.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;sm.codeSize=raw.size();sm.pCode=code.data();VkShaderModule module{};ck(vkCreateShaderModule(device,&sm,nullptr,&module),"shader module");VkPipelineShaderStageCreateInfo stage{};stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;stage.module=module;stage.pName="main";VkComputePipelineCreateInfo cp{};cp.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;cp.stage=stage;cp.layout=pipeline_layout;VkPipeline pipeline{};const auto result=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&cp,nullptr,&pipeline);vkDestroyShaderModule(device,module,nullptr);ck(result,"compute pipeline");pipelines[o.target]=pipeline;}
   VkDescriptorSetAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;ai.descriptorPool=descriptor_pool;ai.descriptorSetCount=1;ai.pSetLayouts=&set_layout;ck(vkAllocateDescriptorSets(device,&ai,&o.set),"allocate descriptors");std::array<VkDescriptorBufferInfo,9>bi{};std::array<VkWriteDescriptorSet,9>writes{};
   for(unsigned j=0;j<9;++j){auto &b=buffers.at(j<o.bindings.size()?o.bindings[j]:"dummy");bi[j]={b.handle,0,b.bytes};auto &w=writes[j];w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w.dstSet=o.set;w.dstBinding=j;w.descriptorCount=1;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&bi[j];}vkUpdateDescriptorSets(device,9,writes.data(),0,nullptr);
  }
 }
 std::vector<double> execute(const Plan&p,unsigned dispatch_count){
  auto cmd=begin();if(query_pool)vkCmdResetQueryPool(cmd,query_pool,0,dispatch_count*2);unsigned qi=0;
  for(const auto&o:p.ops){if(o.kind==Op::Fill)vkCmdFillBuffer(cmd,buffers.at(o.target).handle,o.offset,o.bytes,o.value);
   else {vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipelines.at(o.target));vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_layout,0,1,&o.set,0,nullptr);if(!o.push.empty())vkCmdPushConstants(cmd,pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,static_cast<std::uint32_t>(o.push.size()*4),o.push.data());if(query_pool)vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,query_pool,qi++);vkCmdDispatch(cmd,o.grid[0],o.grid[1],o.grid[2]);if(query_pool)vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,query_pool,qi++);}
   barrier(cmd);
  }submit(cmd);
  std::vector<double>ms(dispatch_count,0);
  if(query_pool){std::vector<std::uint64_t>q(dispatch_count*2);ck(vkGetQueryPoolResults(device,query_pool,0,dispatch_count*2,q.size()*8,q.data(),8,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT),"query results");const auto mask=timestamp_bits>=64?~std::uint64_t(0):(std::uint64_t(1)<<timestamp_bits)-1;for(unsigned i=0;i<dispatch_count;++i)ms[i]=double((q[2*i+1]-q[2*i])&mask)*double(prop.limits.timestampPeriod)*1e-6;}return ms;
 }
 void dump(const Plan&p){for(const auto&spec:p.dumps){auto &b=buffers.at(spec.first);auto s=allocate(b.bytes,true);
  try{auto cmd=begin();barrier(cmd);VkBufferCopy c{0,0,b.bytes};vkCmdCopyBuffer(cmd,b.handle,s.handle,1,&c);VkMemoryBarrier mb{};mb.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;mb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;mb.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&mb,0,nullptr,0,nullptr);submit(cmd);void *ptr=nullptr;ck(vkMapMemory(device,s.memory,0,VK_WHOLE_SIZE,0,&ptr),"map readback");if(!s.coherent){VkMappedMemoryRange mr{};mr.sType=VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;mr.memory=s.memory;mr.size=VK_WHOLE_SIZE;ck(vkInvalidateMappedMemoryRanges(device,1,&mr),"invalidate readback");}std::ofstream f(local_file(p.root,spec.second),std::ios::binary);if(!f)throw std::runtime_error("cannot create output");f.write(reinterpret_cast<const char*>(ptr),static_cast<std::streamsize>(b.bytes));if(!f)throw std::runtime_error("write output failed");vkUnmapMemory(device,s.memory);}catch(...){destroy(s);throw;}destroy(s);
 }}
};
static void list_devices(){VkApplicationInfo app{};app.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;app.apiVersion=VK_API_VERSION_1_0;VkInstanceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ci.pApplicationInfo=&app;VkInstance inst{};ck(vkCreateInstance(&ci,nullptr,&inst),"create list instance");std::uint32_t n=0;auto r=vkEnumeratePhysicalDevices(inst,&n,nullptr);if(r!=VK_SUCCESS){vkDestroyInstance(inst,nullptr);ck(r,"enumerate devices");}std::vector<VkPhysicalDevice>d(n);ck(vkEnumeratePhysicalDevices(inst,&n,d.data()),"enumerate devices");for(unsigned i=0;i<n;++i){VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(d[i],&p);std::cout<<i<<": "<<p.deviceName<<"\n";}vkDestroyInstance(inst,nullptr);}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--list"){list_devices();return 0;}
 if(argc<3){std::cerr<<"Usage: maple-tq2-vk run.plan shader-directory [--device N] [--warmup N] [--repeat N] [--memory-limit-mib N] [--validation]\n       maple-tq2-vk --list\n";return 2;}
 unsigned device=0,warmup=2,repeat=3;std::uint64_t limit=1024ull*1024*1024;bool validation=false;
 for(int i=3;i<argc;++i){std::string flag=argv[i];if(flag=="--validation"){validation=true;continue;}if(i+1>=argc)throw std::runtime_error("missing option value");const auto v=std::stoull(argv[++i]);if(v>std::numeric_limits<unsigned>::max())throw std::runtime_error("option value too large");if(flag=="--device")device=static_cast<unsigned>(v);else if(flag=="--warmup")warmup=static_cast<unsigned>(v);else if(flag=="--repeat")repeat=static_cast<unsigned>(v);else if(flag=="--memory-limit-mib")limit=v*1024*1024;else throw std::runtime_error("unknown option "+flag);}
 if(!repeat||repeat>1000||warmup>1000)throw std::runtime_error("invalid repeat/warmup count");auto plan=load_plan(argv[1]);
 {std::error_code ec;fs::remove(plan.root/"execution.json",ec);}
 // Remove only this plan's declared outputs, so a failed GPU run cannot appear to pass on stale data.
 for(const auto &d:plan.dumps){std::error_code ec;fs::remove(local_file(plan.root,d.second),ec);if(ec)throw std::runtime_error("cannot remove stale output: "+ec.message());}
 const auto dispatches=static_cast<unsigned>(std::count_if(plan.ops.begin(),plan.ops.end(),[](const Op&o){return o.kind==Op::Dispatch;}));if(!dispatches||dispatches>10000)throw std::runtime_error("invalid dispatch count");
 Context c;c.init(device,plan.dot,limit,dispatches,dispatches*2,validation);c.load_buffers(plan);c.prepare(plan,argv[2]);
 for(unsigned i=0;i<warmup;++i)c.execute(plan,dispatches);std::vector<double>totals(dispatches,0);
 const auto start=std::chrono::steady_clock::now();for(unsigned i=0;i<repeat;++i){auto times=c.execute(plan,dispatches);for(unsigned j=0;j<dispatches;++j)totals[j]+=times[j];}
 const auto wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeat;
 c.dump(plan);{std::ofstream receipt(plan.root/"execution.json");receipt<<"{\"backend\":\"Vulkan replay\",\"vulkan_tested\":true,\"native_integer_dot_tested\":"<<(plan.dot?"true":"false")<<"}\n";}std::ofstream csv(plan.root/"timing.csv");csv<<"stage,kernel,mean_ms,timestamp_valid\n";unsigned j=0;for(const auto&o:plan.ops)if(o.kind==Op::Dispatch){const auto ms=totals[j]/repeat;csv<<j<<','<<o.target<<','<<std::setprecision(9)<<ms<<','<<(c.query_pool!=VK_NULL_HANDLE)<<'\n';std::cout<<j<<" "<<o.target<<" "<<ms<<" ms\n";++j;}
 std::cout<<"Replay wall (includes recording/submission/wait, excludes uploads/readback): "<<wall<<" ms/iteration\nPayload: "<<c.allocated_payload/1048576.0<<" MiB\nExecution finished; run check_fixture.py to validate. NOT a model-level pass.\n";return 0;
 }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
