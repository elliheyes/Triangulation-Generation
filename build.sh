mkdir build
cd build && cmake -DDEBUG_MODE=ON .. && make -j
mv libpy_polytope.so py_polytope.so
