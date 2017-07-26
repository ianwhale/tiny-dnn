#pragma once

#include <memory>
#include "tiny_dnn/evo/params.h"



namespace tiny_dnn {

typedef std::shared_ptr<std::vector<float_t>> vec_ptr;

    class Individual {
    public:
        Individual(vec_ptr genome) : mGenome(genome) {}

        vec_ptr getGenome() { return mGenome; }
        void setFitness(float fitness) { mFitness = fitness; }
        float getFitness() { return mFitness; }
    private:
        float mFitness;
        vec_ptr mGenome;
    };
}
