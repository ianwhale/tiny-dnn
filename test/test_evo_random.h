#pragma once

#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(random, dispatcher) {
    auto dispatcher = RandomDispatcher::instance();
    dispatcher->setSeed(42);
    dispatcher->initialize();

    auto gen = dispatcher->getRandom();

    float random;
    for (size_t i = 0; i < 10000; i++) {
        random = gen->getDouble();
        EXPECT_TRUE(0.0f <= random && random < 1.0f);
    }
}

}
