#pragma once

#include <memory>
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(EvoIndividualTest, asexual_reproduction) {
    Random * random = new Random(42);

    Individual ind(100, random);
    auto child = ind.createOffspring(1.0f, 1.0f); // Always mutate.

    auto parent_genome = ind.getGenome();
    auto child_genome = child->getGenome();
    for (size_t i = 0; i < ind.getSize(); i++) {
        EXPECT_NEAR((*parent_genome)[i], (*child_genome)[i], 1.0f);
    }

    delete random;
}

TEST(EvoIndividualTest, sexual_reproduction) {
    Random * random = new Random(42);

    Individual ind1(100, random);
    std::shared_ptr<Individual> ind2 = std::make_shared<Individual>(100, random);

    auto child = ind1.createOffspring(ind2);

    auto ind1_genome = ind1.getGenome();
    auto ind2_genome = ind2->getGenome();
    auto child_genome = child->getGenome();

    for (size_t i = 0; i < ind1.getSize(); i++) {
        EXPECT_TRUE((fabs((*ind1_genome)[i] - (*child_genome)[i]) < 1e-5)
                    || (fabs((*ind1_genome)[i] - (*child_genome)[i]) < 1e-5));
    }

    delete random;
}


}
