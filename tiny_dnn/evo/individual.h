#pragma once

#include <limits>
#include <memory>
#include <iostream>
#include "tiny_dnn/evo/params.h"
#include "tiny_dnn/util/random.h"
#include "tiny_dnn/evo/evolver.h"

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

                mFitness = std::numeric_limits<float>::min();
            }
        }

        /**
         * Copy constructor.
         * @param other
         */
        Individual(const Individual & other) {
            mSize = other.getSize();

            auto genome_ptr = other.getGenome();

            mGenome.empty(); // Sanity check.

            for (float_t weight : *genome_ptr) {
                mGenome.push_back(weight);
            }
        }

        /**
         * Asexual reproduction. Point mutate weights and return child.
         * @return pointer to child
         */
        std::shared_ptr<Individual> createOffspring(float mutation_power,
                                                    float mutation_rate) {
            auto child = std::make_shared<Individual>(*this);

            auto child_genome = child->getGenome();

            for_i(true, child_genome->size(), [&](size_t i) {
                if (uniform_rand(0, 1) < mutation_rate) {
                    (*child_genome)[i] += uniform_rand(-1 * mutation_power,
                                                       mutation_power);
                }
            });

            child->setFitness(mFitness);

            return child;
        }

        /**
         * Sexual reproduction. Crossover and return child.
         * @param  parent
         * @return pointer to child
         */
        std::shared_ptr<Individual> createOffspring(
                                std::shared_ptr<Individual> parent) {
            auto child = std::make_shared<Individual>(*this);
            auto child_genome = child->getGenome();
            auto parent_genome = parent->getGenome();

            for_i(true, child_genome->size(), [&](size_t i) {
                if (uniform_rand(0, 1) < 0.5) {
                    (*child_genome)[i] = (*parent_genome)[i];
                }
            });

            child->setFitness((mFitness + parent->getFitness()) / 2);

            return child;
        }

        /**
         * Get a pointer to the individual's genome
         * @return pointer
         */
        vec_ptr getGenome() const {
            return std::make_shared<std::vector<float_t>>(mGenome);
        }

        /**
         * Only for testing...
         * @param genome
         */
        void setGenome(std::vector<float_t> genome) { mGenome = genome; }

        size_t getSize() const { return mSize; }

        void setFitness(float_t fitness) { mFitness = fitness; }
        float getFitness() { return mFitness; }
    private:
        float mFitness;
        size_t mSize;
        std::vector<float_t> mGenome;
    };
}
