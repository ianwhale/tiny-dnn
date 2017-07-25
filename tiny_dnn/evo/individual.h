#pragma once

#include "tiny_dnn/evo/params.h"

namespace tiny_dnn {
    class Individual {
    public:
        void setFitness(float fitness) { mFitness = fitness; }
        float getFitness() { return mFitness; }
    private:
        float mFitness;
        std::vector<float_t> mGenome;
    };
}
