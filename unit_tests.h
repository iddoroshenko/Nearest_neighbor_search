//
// Created by ilya on 14.01.2021.
//

#ifndef NEAREST_NEIGHBOR_SEARCH_UNIT_TESTS_H
#define NEAREST_NEIGHBOR_SEARCH_UNIT_TESTS_H
#include <tut/tut.hpp>
#include <tut/tut_reporter.hpp>

namespace tut
{
    struct basic{};
    typedef test_group<basic> factory;
    typedef factory::object object;
}

namespace
{
    tut::factory tf("basic test");
}

namespace tut
{
    template<>
    template<>
    void object::test<1>()
    {
        ensure_equals(2+2, 4);
    }
}
#endif //NEAREST_NEIGHBOR_SEARCH_UNIT_TESTS_H
