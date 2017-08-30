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
                auto gen = random_generator::get_instance()();
                std::uniform_real_distribution<float> dst(
                    -1 * Params::initial_weights_delta,
                    Params::initial_weights_delta
                );

                mGenome.push_back(dst(gen));
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

            for_i(true, mSize, [&](size_t i) {
                mGenome[i] = (*genome_ptr)[i];
            });
        }

        /**
         * Asexual reproduction. Point mutate weights and return child.
         * @return pointer to child
         */
        std::shared_ptr<Individual> createOffspring(float mutation_power,
                                                    float mutation_rate) {
            auto child = std::make_shared<Individual>(*this);

            auto child_genome = child->getGenome();

            auto coin_flip = random_generator::get_instance()();
            std::uniform_real_distribution<float> coin_dst(0, 1);

            auto gen = random_generator::get_instance()();
            std::uniform_real_distribution<float> dst(
                -1 * mutation_power,
                mutation_power
            );

            for_i(true, child_genome->size(), [&](size_t i) {
                if (coin_dst(coin_flip) < mutation_rate) {
                    (*child_genome)[i] += dst(gen);
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

            auto coin_flip = random_generator::get_instance()();
            std::uniform_real_distribution<float> coin_dst(0, 1);

            for_i(true, child_genome->size(), [&](size_t i) {
                if (coin_dst(coin_flip) < 0.5) {
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
