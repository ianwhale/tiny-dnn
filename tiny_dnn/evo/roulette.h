#pragma once

#include <memory>
#include <algorithm>
#include <iostream>
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
     * This wheel assumes that all values will be negative.
     * @param individuals
     */
    Roulette(const std::vector<std::shared_ptr<Individual>> & individuals) {
        mTotal = 0.0;
        mSize = individuals.size();
        float_t min = std::numeric_limits<float>::max();
        float_t max = std::numeric_limits<float>::min();
        float_t fitness;
        for (auto individual : individuals) {
            fitness = -1 * individual->getFitness();
            mTotal += fitness;
            max = (fitness > max) ? fitness : max;
            min = (fitness < min) ? fitness : min;
            assert(individual->getFitness() <= 0.0); // No positive fitnesses.
        }

        for (auto individual : individuals) {
            // The adjusted fitness scheme simply turns the negative values into
            // the same layout as one would expect in normal roulette selection.
            mAdjustedFits.push_back(individual->getFitness() + min + max);
        }

        for (float_t adj_fit : mAdjustedFits) {
            std::cout << "Adjusted Fitness: " << adj_fit << std::endl;
        }
    }

    /**
     * Perform a roulette wheel spin.
     * @return size_t i, the index chosen from the fitnesses.
     */
    size_t spin() {
        float_t r_val = uniform_rand(0.0, 1.0) * mTotal;

        float_t sum = 0.0;
        for (size_t i = 0; i < mSize; i++) {
            sum += mAdjustedFits[i];

            if (sum > r_val) {
                return i;
            }
        }

        // Gaurd for rounding error.
        return mSize - 1;
    }

private:
    float_t mTotal;
    size_t mSize;
    std::vector<float_t> mAdjustedFits = {}; //< Adjusted fitnesses.
};

}
