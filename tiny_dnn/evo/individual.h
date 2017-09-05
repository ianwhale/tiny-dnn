#pragma once

#include <limits>
#include <memory>
#include <iostream>
#include "tiny_dnn/evo/params.h"
#include "tiny_dnn/util/random.h"
#include "tiny_dnn/evo/evolver.h"
#include "tiny_dnn/evo/random.h"

namespace tiny_dnn {

typedef std::shared_ptr<std::vector<float_t>> vec_ptr;

    class Individual {
    public:
        /**
         * Generate an initial random genome based on the specified size.
         * @param size
         */
        Individual(size_t size, Random * random) : mSize(size) {
            mRandom = random;

            for (size_t i = 0; i < size; i++) {
                // Randomly initialize genome.
                mGenome.push_back(random->getDouble(
                    -1 * Params::initial_weights_delta,
                    Params::initial_weights_delta
                ));

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

            mGenome.resize(mSize);

            for (size_t i = 0; i < mSize; i++) {
                mGenome[i] = (*genome_ptr)[i];
            }

            // for_i(true, mSize, [&](size_t i) {
            //     mGenome[i] = (*genome_ptr)[i];
            // });

            mRandom = other.getRandom();
        }

        /**
         * Asexual reproduction. Point mutate weights and return child.
         * @return pointer to child
         */
        std::shared_ptr<Individual> createOffspring(float mutation_power,
                                                    float mutation_rate) {
            auto child = std::make_shared<Individual>(*this);
            auto child_genome = child->getGenome();

            for (size_t i = 0; i < child_genome->size(); i++) {
                if (mRandom->getDouble() < mutation_rate) {
                    (*child_genome)[i] +=
                        mRandom->getDouble(-1 * mutation_power, mutation_power);
                }
            }

            // for_i(true, child_genome->size(), [&](size_t i) {
            //     if (mRandom->getDouble() < mutation_rate) {
            //         (*child_genome)[i] +=
            //             mRandom->getDouble(-1 * mutation_power, mutation_power);
            //     }
            // });

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

            for (size_t i = 0; i < child_genome->size(); i++) {
                if (mRandom->getDouble() < 0.5) {
                    (*child_genome)[i] = (*parent_genome)[i];
                }
            }

            // for_i(true, child_genome->size(), [&](size_t i) {
            //     if (mRandom->getDouble() < 0.5) {
            //         (*child_genome)[i] = (*parent_genome)[i];
            //     }
            // });

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
        Random * getRandom() const { return mRandom; }
    private:
        size_t mSize;
        Random * mRandom;
        float mFitness;
        std::vector<float_t> mGenome;
    };
}
