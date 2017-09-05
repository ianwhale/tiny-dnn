#pragma once

#include <iostream>
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(EvoRandomTest, get_double) {
    Random * random = new Random(42);

    float r;
    for (size_t i = 0; i < 100; i++) {
        r = random->getDouble();

        EXPECT_TRUE(0.0f <= r && r < 1.0f);
    }

    delete random;
}

}
