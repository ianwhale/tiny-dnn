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
    template <typename NetType, typename Error>
    class Evolver {
    public:

        /**
         * Utility class to take care of dipatching minibatches.
         * @param labels training labels.
         * @param data training data.
         */
        struct MiniBatchHandler {
            MiniBatchHandler(std::shared_ptr<std::vector<vec_t>> labels,
                             std::shared_ptr<std::vector<vec_t>> data) :
                             mTrainLabels(labels),
                             mTrainData(data)
                             { }

            /**
             * Copy the next mini batch into the provided vector pointers.
             * @param mini_data
             * @param mini_label
             */
            inline void nextBatch(std::vector<vec_t> *mini_labels,
                                  std::vector<vec_t> *mini_data) {
                mini_labels->resize(Params::sample_count);
                mini_data->resize(Params::sample_count);

                for (size_t i = 0; i < Params::sample_count; i++) {
                    if (mIndex >= mTrainLabels->size()) {
                        // We've rolled into the new epoch.
                        mEpoch++;
                        mIndex = 0;
                    }

                    (*mini_labels)[i] = (*mTrainLabels)[mIndex];
                    (*mini_data)[i] = (*mTrainData)[mIndex];
                    mIndex++;
                }
            }

            size_t getEpoch() { return mEpoch; }

        private:
            size_t mIndex = 0;
            size_t mEpoch = 0;
            std::shared_ptr<std::vector<vec_t>> mTrainLabels;
            std::shared_ptr<std::vector<vec_t>> mTrainData;
        };


        /**
         * Evolver constructor.
         * @param nn, network being evolved.
         * @param train_labels, one-hot encodings.
         * @param train_data, some data!
         * @param initialize, should we initalize the population?
         *                    If no, be sure to on your own!
         */
        Evolver(std::shared_ptr<network<NetType>> nn,
                std::shared_ptr<std::vector<vec_t>> train_labels,
                std::shared_ptr<std::vector<vec_t>> train_data)
                : mNetwork(nn), mHandler(train_labels, train_data) {

            for (size_t i = 0; i < Params::population_size; i++) {
                mPopulation.push_back(nullptr);
            }

            mMutationPower = Params::mutation_power;
            mMutationDecayRate = Params::mutation_rate_decay;
            mDecayRate = pow(1 - Params::mutation_power_decay,
                            1.0f / Params::max_generations);
            mRateDecayRate = pow(1 - Params::mutation_rate_decay,
                            1.0f / Params::max_generations);
            mMutationRate = Params::mutation_rate;

            mWeightCount = calculateWeightCount();
            initializePopulation(mWeightCount);
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
        void evolve() {
            while (mCurrentGeneration < Params::max_generations) {
                sortPopulation();
                printInfo();
                reproducePopulation();

                evaluatePopulation();
                mCurrentGeneration++;

                mMutationPower *= mDecayRate;
                mMutationRate *= mRateDecayRate;
            }
        }

        /**
         * Print out some info for the current state of the run.
         */
        void printInfo() {
            std::cout << "Best of generation "
                    << mCurrentGeneration
                    << ": " << mPopulation[0]->getFitness()
                    << std::endl;

            std::cout << "Average of generation "
                    << mCurrentGeneration
                    << ": " << getAverageFitness()
                    << std::endl;

            std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - -"
                      << std::endl;
        }

        /**
         * Assign a fitness to each Individual by evaluating the network and
         * getting the value of the loss function for a minibatch.
         */
        void evaluatePopulation() {
            // TODO: Parallelize! (Worth noting that I'm guessing that
            // the network evaluation routine is not threadsafe...)
            std::vector<vec_t> mini_data;
            std::vector<vec_t> mini_labels;

            mHandler.nextBatch(&mini_labels, &mini_data);

            // Idea to parallelize:
            // Maintain a list of networks (4) and evaluate the first
            // 25% or 12.5% of the population on the first network, etc etc.
            float_t fitness;
            for (auto individual : mPopulation) {
                loadWeights(individual);
                fitness = mNetwork->template get_loss<Error>(mini_data, mini_labels);
                individual->setFitness(individual->getFitness() * (1.0 - Params::fitness_decay_rate)
                            + fitness);
            }
        }

        /**
         * Sort the population by fitness.
         */
        inline void sortPopulation() {
            std::sort(mPopulation.begin(), mPopulation.end(),
                    [](const std::shared_ptr<Individual> a,
                       const std::shared_ptr<Individual> b) -> bool
                   {
                       return a->getFitness() < b->getFitness();
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
         * Average fitness of the population.
         * @return float_t
         */
        float_t getAverageFitness() {
            float_t sum(0);
            for (auto individual : mPopulation) {
                sum += individual->getFitness();
            }
            return sum / mPopulation.size();
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

        /**
         * Get pointer to the minibatch handler (testing).
         * @return pointer to handler.
         */
        const std::shared_ptr<MiniBatchHandler> getMiniBatchHandler() const {
            return std::make_shared<MiniBatchHandler>(mHandler);
        }
    protected:
        int mCurrentGeneration = 0;
        std::vector<std::shared_ptr<Individual>> mPopulation;
        std::shared_ptr<network<NetType>> mNetwork;

        size_t mWeightCount;

        float mMutationPower;
        float mMutationDecayRate;
        float mMutationRate;
        float mDecayRate;
        float mRateDecayRate;

        MiniBatchHandler mHandler;
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

            evaluatePopulation();
        }
    };

}
