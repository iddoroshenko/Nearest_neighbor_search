#include "tests.h"
#include "unit_tests.h"

void call_unitTests() {
    tut::reporter reporter;
    tut::test_runner_singleton::get().set_callback(&reporter);

    tut::test_runner_singleton::get().run_tests();
}

int main() {
    srand(time(nullptr));
    call_unitTests();
    //sift_test1B();
/*
    Points points;
    std::vector<std::vector<int>> queries;
    std::vector<std::vector<int>> answers;
    read_sift_test1B(points,queries,answers);*/
}
