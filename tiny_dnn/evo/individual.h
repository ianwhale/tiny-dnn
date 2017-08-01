#pragma once

#include <memory>
#include "tiny_dnn/evo/params.h"
#include "tiny_dnn/util/random.h"

namespace tiny_dnn {

typedef std::shared_ptr<std::vector<float_t>> vec_ptr;

    class Individual {
    public:
        /**
         * Generate an initial random genome based on the specified size.
         * @param size
         */
        Individual(size_t size) : mSize(size) {
            for (size_t i = 0; i < size; i++) {
                // Randomly initialize genome.
                mGenome.push_back(
                    uniform_rand(-1 * Params::initial_weights_delta,
                                 Params::initial_weights_delta)
                );
            }
        }

        vec_ptr getGenome() {
            return std::make_shared<std::vector<float_t>>(mGenome);
        }
        void setGenome(std::vector<float_t> genome) { mGenome = genome; }

        void setFitness(float_t fitness) { mFitness = fitness; }
        float_t getFitness() { return mFitness; }
    private:
        float_t mFitness;
        size_t mSize;
        std::vector<float_t> mGenome;
    };
}
