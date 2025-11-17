add_executable(llamax_simple simple.cpp)
target_link_libraries(llamax_simple llamax)

add_executable(llamax_chat chat.cpp)
target_link_libraries(llamax_chat llamax)

add_executable(llamax_grammar grammar.cpp)
target_link_libraries(llamax_grammar llamax)
