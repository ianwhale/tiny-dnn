/**
 * This modified from the (fantastic) Empirical library.
 * Find the original source at:
 *  https://github.com/devosoft/Empirical/blob/master/source/tools/Random.h
 */

#pragma once

#include<iostream>
#include <memory>
#include <random>

namespace tiny_dnn {
    class Random {
    public:
        /**
        * Set up the random generator object.
        * @param _seed The seed of the random number generator.
        * A negative seed means that the random number generator gets its
        * seed from the actual system time.
        **/
        Random(const int _seed = 0) : seed(_seed), inext(0),
                                      inextp(0), expRV(0) {
            for (int i = 0; i < 56; ++i) ma[i] = 0;
            init();
        }

        ~Random() {;}

        /**
         * Set the seed for the random generator.
         * @param s integer seed value.
         */
        inline void setSeed(const int s) {
            seed = s;
            init();
        }

        /**
         * Get the current seed.
         * @return the random seed.
         */
        inline int getSeed() { return seed; }

        /**
         * Generate a double between 0.0 and 1.0
         *
         * @return The pseudo random number.
         **/
        inline double getDouble() { return get() / (double) _RAND_MBIG; }

        /**
         * Generate a double out of a given interval.
         *
         * @return The pseudo random number.
         * @param min The lower bound for the random numbers.
         * @param max The upper bound for the random numbers (will never be returned).
         **/
        inline double getDouble(const double min, const double max) {
          return getDouble() * (max - min) + min;
        }

        /**
         * Generate an uint32_t.
         *
         * @return The pseudo random number.
         * @param max The upper bound for the random numbers (will never be returned).
         **/
        template <typename T>
        inline uint32_t getUInt(const T max) {
          return static_cast<uint32_t>(getDouble() * static_cast<double>(max));
        }

        /**
         * Generate an int out of an interval.
         *
         * @return The pseudo random number.
         * @param min The lower bound for the random numbers.
         * @param max The upper bound for the random numbers (will never be returned).
         **/
        inline int getInt(const int max) {
            return static_cast<int>(getUInt((uint32_t) max));
        }
        inline int getInt(const int min, const int max) {
            return getInt(max - min) + min;
        }

    protected:
        int seed;
        int inext;
        int inextp;
        int ma[56];

        double expRV;

        static const int32_t _RAND_MBIG = 1000000000;
        static const int32_t _RAND_MSEED = 161803398;

        // Setup, called on initialization.
        void init()
        {
          // Clear variables
          for (int i = 0; i < 56; ++i) ma[i] = 0;

          int32_t mj = (_RAND_MSEED - seed) % _RAND_MBIG;
          ma[55] = mj;
          int32_t mk = 1;

          for (int32_t i = 1; i < 55; ++i) {
            int32_t ii = (21 * i) % 55;
            ma[ii] = mk;
            mk = mj - mk;
            if (mk < 0) mk += _RAND_MBIG;
            mj = ma[ii];
          }

          for (int32_t k = 0; k < 4; ++k) {
            for (int32_t j = 1; j < 55; ++j) {
              ma[j] -= ma[1 + (j + 30) % 55];
              if (ma[j] < 0) ma[j] += _RAND_MBIG;
            }
          }

          inext = 0;
          inextp = 31;

          // Setup variables used by Statistical Distribution functions
          expRV = -log(Random::get() / (double) _RAND_MBIG);
        }

        // Basic Random number
        // Returns a random number [0,_RAND_MBIG)
        int32_t get() {
          if (++inext == 56) inext = 0;
          if (++inextp == 56) inextp = 0;

          int mj = ma[inext] - ma[inextp];
          if (mj < 0) mj += _RAND_MBIG;

          ma[inext] = mj;

          return mj;
        }
    };
}
