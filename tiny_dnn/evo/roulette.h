#pragma once

#include <memory>
#include <algorithm>
#include <cmath>
#include <limits>
#include <assert.h>
#include "tiny_dnn/evo/individual.h"
#include "tiny_dnn/util/random.h"

namespace tiny_dnn {

/**
 * Roulette wheel class for roulette selection.
 * *Note*: since this roulette wheel is being used for minimizing loss functions
 * it will assume that lower fitnesses are better.
 */
class Roulette {

public:
    /**
     * This wheel assumes that all values will be positive and a minimization
     * scheme. .
     * @param individuals
     */
    Roulette(const std::vector<std::shared_ptr<Individual>> & individuals) {
        mSize = individuals.size();

        float_t min = std::numeric_limits<float_t>::max();
        float_t max = std::numeric_limits<float_t>::min();
        float_t fitness;

        for (auto individual : individuals) {
            fitness = individual->getFitness();
            max = (fitness > max) ? fitness : max;
            min = (fitness < min) ? fitness : min;
        }

        float_t adjustment =  max + min;

        for (auto individual : individuals) {
            // Adjust the fitnesses so the smallest value is the biggest value.
            fitness = adjustment - individual->getFitness();
            mTotal += fitness;
            mAdjustedFits.push_back(fitness);
        }

        mDistribution = std::uniform_real_distribution<float>(0, mTotal);
    }

    /**
     * Perform a roulette wheel spin.
     * @return size_t i, the index chosen from the fitness distribution.
     */
    size_t spin() {
        float_t r_val = mDistribution(random_generator::get_instance()());

        for (size_t i = 0; i < mSize; i++) {
            r_val -= mAdjustedFits[i];

            if (r_val <= 0) {
                return i;
            }
        }

        // Gaurd for rounding error.
        return mSize - 1;
    }

private:
    std::uniform_real_distribution<float> mDistribution;
    float_t mTotal = 0.0;
    size_t mSize = 0;
    std::vector<float_t> mAdjustedFits = {}; //< Adjusted fitnesses.
};

}
