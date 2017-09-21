#pragma once

#include <array>
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <utility>
#include <thread>
#include <limits>
#include "tiny_dnn/evo/params.h"
#include "tiny_dnn/evo/individual.h"
#include "tiny_dnn/evo/roulette.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

typedef std::shared_ptr<std::vector<std::shared_ptr<Individual>>> population_t;

    /**
     * Evolver class. Aims to minimize the weights of the network using LEEA.
     */
    template <typename Error, size_t N>
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
         * @param networks, multiple of the same network for parallelization.
         * @param train_labels, one-hot encodings.
         * @param train_data, some data!
         * @param initialize, should we initalize the population?
         *                    If no, be sure to on your own!
         */
        Evolver(std::array<std::shared_ptr<network<sequential>>, N> * networks,
                std::shared_ptr<std::vector<vec_t>> train_labels,
                std::shared_ptr<std::vector<vec_t>> train_data,
                Random * random)
                : mNetworks(networks), mHandler(train_labels, train_data) {
            mRandom = random;

            for (size_t i = 0; i < Params::population_size; i++) {
                mPopulation.push_back(nullptr);
                mGenerationErrors.push_back(0.0);
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
        std::shared_ptr<std::vector<float>> getCurrentNetworkWeights(size_t idx) {
            auto network_weights = std::make_shared<std::vector<float>>();
            for (auto & layer : *((*mNetworks)[idx])) {
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
        void loadWeights(std::shared_ptr<Individual> individual, size_t id) {
            int idx = 0;
            for (auto & layer : *((*mNetworks)[id])) {
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
            std::cout << "Best fitness of generation "
                    << mCurrentGeneration
                    << ": " << mPopulation[0]->getFitness()
                    << std::endl;

            std::cout << "Average fitness of generation "
                    << mCurrentGeneration
                    << ": " << getAverageFitness()
                    << std::endl;

            float_t lowest_error = std::numeric_limits<float_t>::max();
            float_t average_error = 0;

            for (float_t error : mGenerationErrors) {
                if (error < lowest_error) {
                    lowest_error = error;
                }
                average_error += error;
            }

            average_error /= mGenerationErrors.size();

            std::cout << "Lowest error of generation "
                    << mCurrentGeneration
                    << ": " << lowest_error
                    << std::endl;

            std::cout << "Average error of generation "
                    << mCurrentGeneration
                    << ": " << average_error
                    << std::endl;

            std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - -"
                      << std::endl;
        }

        /**
         * Assign a fitness to each Individual by evaluating the network and
         * getting the value of the loss function for a minibatch.
         */
        void evaluatePopulation() {
            std::vector<vec_t> mini_data;
            std::vector<vec_t> mini_labels;

            mHandler.nextBatch(&mini_labels, &mini_data);

            // Inspired by code from tiny_dnn/util/parallel_for.h
            size_t range_size = mPopulation.size() / N;

            if (range_size * N < (mPopulation.size())) {
                range_size++;
            }

            size_t begin = 0;
            size_t end = begin + range_size;
            std::vector<std::thread> threads;

            for (size_t i = 0; i < N; i++) {
                threads.push_back(
                    std::move(
                        std::thread(
                            [this, begin, end, i, mini_data, mini_labels] {
                                this->evaluateRange(
                                    begin, end, i, mini_data, mini_labels
                                );
                            }
                        )
                    )
                );

                begin += range_size;
                end = begin + range_size;
                if (begin >= mPopulation.size()) {
                    break;
                }

                if (end > mPopulation.size()) {
                    end = mPopulation.size();
                }
            }

            for (auto &thread_ : threads) {
                thread_.join();
            }
        }

        /**
         * Evaluates a range of the population using a particula
         * @param start starting index of evaluation.
         * @param end   ending index of evaluation.
         * @param id    which network to evaluate with.
         * @param mini_data copy of data to use.
         * @param mini_labels copy of labels to use.
         */
        void evaluateRange(size_t start, size_t end, size_t id,
            std::vector<vec_t> mini_data, std::vector<vec_t> mini_labels) {

            float fitness = mini_data.size();
            float previous_fitness;
            float_t error;
            for (size_t i = start; i < end; i++) {
                previous_fitness = mPopulation[i]->getFitness();
                previous_fitness *= 1 - Params::fitness_decay_rate;

                loadWeights(mPopulation[i], id);
                error = (*mNetworks)[id]->template get_loss<Error>(mini_data, mini_labels);
                fitness -= error;
                mGenerationErrors[i] = error;

                fitness = (fitness < Params::min_fitness) ? Params::min_fitness
                                                          : fitness;

                mPopulation[i]->setFitness(fitness + previous_fitness);
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

            Roulette wheel(top_individuals, mRandom);

            size_t index;
            for (size_t i = 0; i < Params::population_size; i++) {
                index = wheel.spin();

                // Should we do sexual reproduction?
                if (mRandom->getDouble() < Params::sex_proportion) {
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
        population_t getPopulation() {
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
        std::vector<float_t> mGenerationErrors;


        std::array<std::shared_ptr<network<sequential>>, N> * mNetworks;

        size_t mWeightCount;

        float mMutationPower;
        float mMutationDecayRate;
        float mMutationRate;
        float mDecayRate;
        float mRateDecayRate;

        MiniBatchHandler mHandler;
        Random * mRandom;
    private:
        /**
         * Calculate how many weights are in the network.
         * @return count
         */
        size_t calculateWeightCount() {
            size_t count = 0;
            for (auto layer : *((*mNetworks)[0])) {
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
            for (size_t i = 0; i < mPopulation.size(); i++) {
                mPopulation[i] = std::make_shared<Individual>(genome_length, mRandom);
            }

            evaluatePopulation();
        }
    };

}
