#include "tests.h"
#include "unit_tests.h"

int main() {
    srand(time(nullptr));
    tut::reporter reporter;
    tut::test_runner_singleton::get().set_callback(&reporter);

    tut::test_runner_singleton::get().run_tests();

    //sift_test1B();
}
