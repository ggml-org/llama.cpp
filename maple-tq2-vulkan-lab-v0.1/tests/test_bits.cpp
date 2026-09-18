#include "tq2_bits.hpp"
#include <iostream>
#include <random>
#include <vector>
int main(){
 using namespace maple_tq2;
 std::array<std::uint8_t,66> b{};
 std::uint64_t checks=0;
 // Every possible byte, every packed four-element position, all bit planes.
 for(unsigned value=0;value<256;++value){
  b.fill(static_cast<std::uint8_t>(value));
  for(std::size_t i=0;i<64;++i){auto p=unpack4(b.data(),i);for(int j=0;j<4;++j){if(signed_lane(p,j)!=decode_code(b.data(),i*4+j))return 1;++checks;}}
 }
 std::mt19937 rng(18092026);
 for(int trial=0;trial<4096;++trial){
  for(auto &v:b)v=static_cast<std::uint8_t>(rng());
  for(std::size_t i=0;i<64;++i){const auto p=unpack4(b.data(),i);const auto a=static_cast<std::uint32_t>(rng());int ref=0;
   for(int j=0;j<4;++j){if(signed_lane(p,j)!=decode_code(b.data(),i*4+j))return 2;ref+=decode_code(b.data(),i*4+j)*signed_lane(a,j);++checks;}
   if(dot4(p,a)!=ref)return 3;
  }
 }
 if(checked_bytes(256,512,2048)!=69206016u)return 4;
 bool caught=false;try{checked_bytes(1,512,255);}catch(const std::invalid_argument&){caught=true;}if(!caught)return 5;
 std::cout<<"PASS: "<<checks<<" unpack checks + 262144 packed dot checks\n";
 return 0;
}
