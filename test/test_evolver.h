#pragma once

#include <memory>
#include <iostream>
#include <algorithm>
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

// Puts 10 random data points into the data pointer.
void put_random_data(std::vector<vec_t> *data, size_t columns, Random * random) {
    data->resize(10);
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < columns; j++) {
            (*data)[i].push_back(random->getDouble(float_t(0), float_t(1)));
        }
    }
}

void make_test_network(std::shared_ptr<network<sequential>> nn) {
    core::backend_t backend_type = core::default_engine();
    *nn << fully_connected_layer(5, 2, true, backend_type)
    << sigmoid_layer()
    << fully_connected_layer(2, 1, true, backend_type);
}

TEST(EvoEvolverTest, initialize_population) {
    Random * random = new Random(42);

    auto nn = std::make_shared<network<sequential>>();
    make_test_network(nn);

    std::array<std::shared_ptr<network<sequential>>, 1> networks;
    networks[0] = nn;

    std::vector<vec_t> train_labels;
    std::vector<vec_t> train_data;

    put_random_data(&train_labels, 1, random);
    put_random_data(&train_data, 5, random);

    Evolver<se, 1> evo(&networks,
                    std::make_shared<std::vector<vec_t>>(train_labels),
                    std::make_shared<std::vector<vec_t>>(train_data),
                    random);

    size_t size = evo.getWeightCount();
    population_t population = evo.getPopulation();

    for (auto individual : *population) {
        auto genome = *(individual->getGenome());
        EXPECT_EQ(size, genome.size());

        for (auto weight : genome) {
            EXPECT_TRUE(weight <= Params::initial_weights_delta
                        && weight >= (-1 * Params::initial_weights_delta));
        }
    }

    delete random;
}

TEST(EvoEvolverTest, load_unload_weights) {
    Random * random = new Random(42);

    auto nn = std::make_shared<network<sequential>>();
    make_test_network(nn);

    std::array<std::shared_ptr<network<sequential>>, 1> networks;
    networks[0] = nn;

    std::vector<vec_t> train_labels;
    std::vector<vec_t> train_data;

    put_random_data(&train_labels, 1, random);
    put_random_data(&train_data, 5, random);

    Evolver<se, 1> evo(&networks,
                    std::make_shared<std::vector<vec_t>>(train_labels),
                    std::make_shared<std::vector<vec_t>>(train_data),
                    random);

    size_t weight_count = evo.getWeightCount();

    std::shared_ptr<Individual> indv_ptr(new Individual(weight_count, random));

    evo.loadWeights(indv_ptr, 0);
    auto network_weights = *(evo.getCurrentNetworkWeights(0));
    auto genome = *(indv_ptr->getGenome());

    for (size_t i = 0; i < weight_count; i++) {
        EXPECT_EQ(genome[i], network_weights[i]);
    }

    genome = *(indv_ptr->getGenome());

    for (auto & weight : genome) {
        weight++;
    }

    indv_ptr->setGenome(genome);

    evo.loadWeights(indv_ptr, 0);
    network_weights = *(evo.getCurrentNetworkWeights(0));

    for (size_t i = 0; i < weight_count; i++) {
        EXPECT_EQ(genome[i], network_weights[i]);
    }

    delete random;
}

TEST(EvoEvolverTest, reproduce_population) {
    Random * random = new Random(42);

    auto nn = std::make_shared<network<sequential>>();
    make_test_network(nn);

    std::array<std::shared_ptr<network<sequential>>, 1> networks;
    networks[0] = nn;

    std::vector<vec_t> train_labels;
    std::vector<vec_t> train_data;

    put_random_data(&train_labels, 1, random);
    put_random_data(&train_data, 5, random);

    Evolver<se, 1> evo(&networks,
                    std::make_shared<std::vector<vec_t>>(train_labels),
                    std::make_shared<std::vector<vec_t>>(train_data),
                    random);

    auto pre_repro = evo.getPopulation();

    EXPECT_NO_THROW(evo.reproducePopulation());

    auto post_repro = evo.getPopulation();

    EXPECT_EQ(pre_repro->size(), post_repro->size());

    // Don't know how to test anything else...
    delete random;
}

TEST(EvoEvolverTest, mini_batch_handler) {
    Random * random = new Random(42);

    auto nn = std::make_shared<network<sequential>>();
    make_test_network(nn);

    std::array<std::shared_ptr<network<sequential>>, 1> networks;
    networks[0] = nn;

    std::vector<vec_t> train_labels;
    std::vector<vec_t> train_data;

    vec_t dummy(5, float_t(0));
    vec_t dummy_label(1, float_t(0));
    // One more than sample count to ensure we get a wrap around.
    for (size_t i = 0; i < Params::sample_count + 1; i++) {
        train_labels.push_back(dummy_label);
        train_data.push_back(dummy);

        // Something to differentiate the samples.
        train_labels[i][0] = i;
        train_data[i][i] = float_t(1);
    }

    Evolver<se, 1> evo(&networks,
                    std::make_shared<std::vector<vec_t>>(train_labels),
                    std::make_shared<std::vector<vec_t>>(train_data),
                    random);

    // The population is evaluated on evolver construction so we should expect
    // a rollover into the next epoch on the this batch.

    std::vector<vec_t> mini_labels;
    std::vector<vec_t> mini_data;

    evo.getMiniBatchHandler()->nextBatch(&mini_labels, &mini_data);

    EXPECT_EQ(train_labels[train_labels.size() - 1], mini_labels[0]);
    EXPECT_EQ(train_data[train_data.size() - 1], mini_data[0]);

    for (size_t i = 1; i < mini_labels.size(); i++) {
        EXPECT_EQ(train_labels[i - 1], mini_labels[i]);
        EXPECT_EQ(train_data[i - 1], mini_data[i]);
    }

    delete random;
}

}
