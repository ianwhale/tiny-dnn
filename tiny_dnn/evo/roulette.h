#pragma once

#include <memory>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include "tiny_dnn/evo/individual.h"

namespace tiny_dnn {

/**
 * Standard roulette wheel for fitness proportionate selection.
 */
class Roulette {

public:
    /**
     * This wheel assumes that all values will be positive.
     * @param individuals
     */
    Roulette(const std::vector<std::shared_ptr<Individual>> & individuals,
             Random * random) {
        mRandom = random;
        mSize = individuals.size();

        float_t total(0.0);
        for (auto individual : individuals) {
            total += individual->getFitness();
        }

        for (auto individual : individuals) {
            mProbDist.push_back(individual->getFitness() / total);
        }
    }

    /**
     * Perform a roulette wheel spin.
     * @return i, the index chosen from the fitness distribution.
     */
    size_t spin() {
        float_t r_val = mRandom->getDouble(0, 1);

        for (size_t i = 0; i < mSize; i++) {
            r_val -= mProbDist[i];

            if (r_val <= 0) {
                return i;
            }
        }

        // Gaurd for rounding error.
        return mSize - 1;
    }

private:
    Random * mRandom;
    size_t mSize = 0;
    std::vector<float_t> mProbDist;
};

}
