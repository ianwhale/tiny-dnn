#pragma once

#include <map>
#include <iostream>
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(roulette, spin) {
    std::vector<float_t> mock_fitnesses = {-30.143, -60.556, -90.334};

    std::vector<std::shared_ptr<Individual>> mock_individuals;
    for (float_t fitness : mock_fitnesses) {
        Individual indv(42);
        indv.setFitness(fitness);
        mock_individuals.push_back(std::make_shared<Individual>(indv));
    }

    std::map<float_t, size_t> counts;

    for (size_t i = 0; i < mock_fitnesses.size(); i++) {
        counts[mock_fitnesses[i]] = 0;
    }

    Roulette wheel(mock_individuals);

    size_t pos;
    for (size_t i = 0; i < 10000; i++) {
        pos = wheel.spin();
        counts[mock_fitnesses[pos]] += 1;
    }

    for (size_t i = 0; i < mock_fitnesses.size() - 1; i++) {
        std::cout << "Count at " << mock_fitnesses[i] << ": " << counts[mock_fitnesses[i]] << std::endl;
        EXPECT_TRUE(counts[mock_fitnesses[i]] > counts[mock_fitnesses[i + 1]]);
    }
    std::cout << "Count at " << mock_fitnesses[mock_fitnesses.size() - 1] << ": "
            << counts[mock_fitnesses[mock_fitnesses.size() - 1]] << std::endl;
}

}
