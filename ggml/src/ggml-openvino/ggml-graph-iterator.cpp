#include "ggml-graph-iterator.h"
#include <ggml.h>
#include <ggml-impl.h>

namespace ov {
namespace frontend {
namespace tensorflow { 
namespace ggml {

GgmlOvGraphIterator::GgmlOvGraphIterator(struct ggml_cgraph * cgraph) 
    :m_cgraph(cgraph) {
    initialize_decoders(); 
    #ifdef GGML_OPENVINO_DEBUG   
        dump_graph_iterator();
    #endif
}

 void GgmlOvGraphIterator::initialize_decoders() {
    auto nodes_size = m_cgraph->n_nodes;
    // Initialize decoder for each node
    // m_decoders.resize(static_cast<size_t>(nodes_size));

    for (int i = 0; i < nodes_size; ++i) {
        // Skip View Op
        if (m_cgraph->nodes[i] ->op == GGML_OP_VIEW || m_cgraph->nodes[i] ->op == GGML_OP_PERMUTE) {
            continue;
        }
        auto decoder = std::make_shared<GgmlOvDecoder>(m_cgraph->nodes[i], m_cgraph);
        m_decoders.push_back(decoder);
        for (size_t inp = 0; inp < decoder->get_input_size(); ++inp) {
            // if (i == 0 || decoder->is_graph_input(inp)) {
                m_input_names.push_back(decoder->get_input_name(inp));
            // }
        }
        for (size_t inp = 0; inp < decoder->get_output_size(); ++inp) {
            if (i == nodes_size - 1 || decoder->is_graph_output(inp)) {
                m_output_names.push_back(decoder->get_output_name(inp));
            }
        }
    }

}

void GgmlOvGraphIterator::reset() {
        node_index = 0;
    }

size_t GgmlOvGraphIterator::size() const  {
    return m_decoders.size();
}

void GgmlOvGraphIterator::next()  {
    node_index++;
}

bool GgmlOvGraphIterator::is_end() const {
    return node_index >= m_decoders.size();
}

std::shared_ptr<DecoderBase> GgmlOvGraphIterator::get_decoder() const {
    return m_decoders[node_index];
}

std::vector<std::string> GgmlOvGraphIterator::get_input_names() const {
    return m_input_names;
}

std::vector<std::string> GgmlOvGraphIterator::get_output_names() const {
    return m_output_names;
}

void GgmlOvGraphIterator::dump_graph_iterator() const {
    for (size_t i = 0; i < m_decoders.size(); ++i) {
        GGML_LOG_INFO("OP %zu: %s\n", i, m_decoders[i]->get_op_name().c_str());
        for (size_t inp = 0; inp < m_decoders[i]->get_input_size(); ++inp) {
            ov::PartialShape pshape = std::dynamic_pointer_cast<GgmlOvDecoder>(m_decoders[i])->get_input_shape(inp);
            ov::element::Type ptype = std::dynamic_pointer_cast<GgmlOvDecoder>(m_decoders[i])->get_input_type(inp); 
            GGML_LOG_INFO("Input name: %s\n", std::dynamic_pointer_cast<GgmlOvDecoder>(m_decoders[i])->get_input_name(inp).c_str());
            GGML_LOG_INFO("Input shape: %s\n", pshape.to_string().c_str());
            GGML_LOG_INFO("Input type: %s\n", ptype.to_string().c_str());
        }
        for (size_t outp = 0; outp < std::dynamic_pointer_cast<GgmlOvDecoder>(m_decoders[i])->get_output_size(); ++outp) {
            ov::PartialShape pshape = std::dynamic_pointer_cast<GgmlOvDecoder>(m_decoders[i])->get_output_shape(outp);
            ov::element::Type ptype = std::dynamic_pointer_cast<GgmlOvDecoder>(m_decoders[i])->get_output_type(outp); 
            GGML_LOG_INFO("Output name: %s\n", std::dynamic_pointer_cast<GgmlOvDecoder>(m_decoders[i])->get_output_name(outp).c_str());
            GGML_LOG_INFO("Output shape: %s\n", pshape.to_string().c_str());
            GGML_LOG_INFO("Output type: %s\n", ptype.to_string().c_str());
            
        }   
    }
}
    
}
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
