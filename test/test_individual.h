#pragma once

#include <memory>
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(individual, asexual_reproduction) {
    Individual ind(100);
    auto child = ind.createOffspring(1.0f, 1.0f); // Always mutate.

    auto parent_genome = ind.getGenome();
    auto child_genome = child->getGenome();
    for (size_t i = 0; i < ind.getSize(); i++) {
        EXPECT_NEAR((*parent_genome)[i], (*child_genome)[i], 1.0f);
        EXPECT_TRUE((*parent_genome)[i] != (*child_genome)[i]);
    }
}

TEST(individual, sexual_reproduction) {
    Individual ind1(100);
    Individual ind2(100);

    auto child = ind1.createOffspring(std::make_shared<Individual>(ind2));

    auto ind1_genome = ind1.getGenome();
    auto ind2_genome = ind2.getGenome();
    auto child_genome = child->getGenome();

    for (size_t i = 0; i < ind1.getSize(); i++) {
        EXPECT_TRUE((fabs((*ind1_genome)[i] - (*child_genome)[i]) < 1e-5)
                    || (fabs((*ind1_genome)[i] - (*child_genome)[i]) < 1e-5));
    }

}


}
