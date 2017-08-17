#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <utility>
#include "tiny_dnn/evo/params.h"
#include "tiny_dnn/evo/individual.h"
#include "tiny_dnn/evo/roulette.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

typedef std::shared_ptr<std::vector<std::shared_ptr<Individual>>> pop_ptr;

    /**
     * Evolver class. Aims to minimize the weights of the network using LEEA.
     */
    template <typename NetType>
    class Evolver {
    public:
        Evolver(std::shared_ptr<network<NetType>> nn,
                std::shared_ptr<std::vector<label_t>> train_labels,
                std::shared_ptr<std::vector<vec_t>> train_data)
                : mNetwork(nn) {

            for (size_t i = 0; i < Params::population_size; i++) {
                mPopulation.push_back(nullptr);
            }

            mWeightCount = calculateWeightCount();

            initializePopulation(mWeightCount);

            mMutationPower = Params::mutation_power;
            mMutationDecayRate = Params::mutation_rate_decay;
            mDecayRate = pow(1 - Params::mutation_power_decay,
                            1.0f / Params::max_generations);
            mRateDecayRate = pow(1 - Params::mutation_rate_decay,
                            1.0f / Params::max_generations);
            mMutationRate = Params::mutation_rate;
        }

        /**
         * Get a pointer to the current network weights.
         * @return shared_ptr
         */
        std::shared_ptr<std::vector<float>> getCurrentNetworkWeights() {
            auto network_weights = std::make_shared<std::vector<float>>();
            for (auto & layer : *mNetwork) {
                for (auto & weights : layer->weights()) {
                    for (float weight : *weights) {
                        network_weights->push_back(weight);
                    }
                }
            }

            return network_weights;
        }

        /**
         * Loads a network's weights with an individual's genome.
         * @param individual
         */
        void loadWeights(std::shared_ptr<Individual> individual) {
            int idx = 0;
            for (auto & layer : *mNetwork) {
                layer->load(*(individual->getGenome()), idx);
            }
        }

        /**
         * Main evolution loop.
         * Evaluate to find best individuals and reproduce best.
         */
        template <typename Error> // Error function to use.
        void evolve() {
            while (mCurrentGeneration < Params::max_generations) {
                evaluatePopulation<Error>();
                sortPopulation();
                reproducePopulation();
                mCurrentGeneration++;

                mMutationPower *= mDecayRate;
                mMutationRate *= mRateDecayRate;
            }
        }

        /**
         * Assign a fitness to each Individual by evaluating the network and
         * getting the value of the loss function for a minibatch.
         */
        template <typename Error>
        void evaluatePopulation() {
            // TODO: Parallelize (Worth noting that I'm guessing that
            // the network evaluation routine is not threadsafe...)

        }

        /**
         * Sort the population by fitness.
         */
        inline void sortPopulation() {
            std::sort(mPopulation.begin(), mPopulation.end(),
                    [](const std::shared_ptr<Individual> a,
                       const std::shared_ptr<Individual> b) -> bool
                   {
                       // Sort in descending order.
                       return a->getFitness() > b->getFitness();
                   });
        }

        /**
         * Reproduce the best individuals based on Params::selection_proportion.
         */
        void reproducePopulation() {
            std::vector<std::shared_ptr<Individual>> newPopulation;

            // At this point, population will (should) be sorted.
            std::vector<std::shared_ptr<Individual>> top_individuals;
            for (size_t i = 0;
                i < (size_t)(Params::population_size * Params::selection_proportion);
                i++) {
                top_individuals.push_back(mPopulation[i]);
            }

            Roulette wheel(top_individuals);

            size_t index;
            for (size_t i = 0; i < Params::population_size; i++) {
                index = wheel.spin();

                // Should we do sexual reproduction?
                if (uniform_rand(0.0, 1.0) < Params::sex_proportion) {
                    newPopulation.push_back(mPopulation[index]->createOffspring(
                        mPopulation[wheel.spin()]));
                }
                else {
                    newPopulation.push_back(mPopulation[index]->createOffspring(
                        mMutationPower, mMutationRate));
                }
            }

            mPopulation = newPopulation;
        }

        /**
         * Mostly for testing or gathering stats.
         * @return shared pointer to the population.
         */
        pop_ptr getPopulation() {
            return std::make_shared<std::vector<std::shared_ptr<Individual>>>
                    (mPopulation);
        }

        /**
         * Get how many weights are in the network.
         * @return mWeightCount
         */
        size_t getWeightCount() { return mWeightCount; }

    protected:
        int mCurrentGeneration = 0;
        std::vector<std::shared_ptr<Individual>> mPopulation;
        std::shared_ptr<network<NetType>> mNetwork;
        std::shared_ptr<std::vector<label_t>> mTrainLabels;
        std::shared_ptr<std::vector<vec_t>> mTrainData;

        size_t mWeightCount;

        // Mutation parameters.
        float mMutationPower;
        float mMutationDecayRate;
        float mMutationRate;
        float mDecayRate;
        float mRateDecayRate;

    private:
        /**
         * Calculate how many weights are in the network.
         * @return count
         */
        size_t calculateWeightCount() {
            size_t count = 0;
            for (auto layer : *mNetwork) {
                for (auto & weights : layer->weights()) {
                    count += weights->size();
                }
            }
            return count;
        }

        /**
         * Initialize the population in parallel.
         * @param genome_length
         */
        void initializePopulation(size_t genome_length) {
            for_i(true, Params::population_size, [&](size_t i) {
                mPopulation[i] = std::make_shared<Individual>(genome_length);
            });
        }
    };

}
