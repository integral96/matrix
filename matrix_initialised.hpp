#pragma once

#define BOOST_THREAD_PROVIDES_FUTURE
#include <iostream>
#include <functional>
#include <utility>
#include <boost/array.hpp>
#include <boost/timer/timer.hpp>
#include <boost/random.hpp>
#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include <boost/thread/future.hpp>

#include <matrix_mult.hpp>

template <size_t N = 3, size_t M = 3>
    using my_matrix = array2d<size_t, N, M, std::vector>;

struct matrix
{
    my_matrix<8, 8>     A_8;
    my_matrix<16, 16>   A_16;
    my_matrix<32, 32>   A_32;
    my_matrix<64, 64>   A_64;
    ///////
    my_matrix<8, 8>     B_8;
    my_matrix<16, 16>   B_16;
    my_matrix<32, 32>   B_32;
    my_matrix<64, 64>   B_64;
    ///////
    my_matrix<8, 8>     C_8;
    my_matrix<16, 16>   C_16;
    my_matrix<32, 32>   C_32;
    my_matrix<64, 64>   C_64;

};

class multyplyMatrix
{
    std::shared_ptr<matrix> ptr_mtrx;
    std::time_t now;
    boost::random::mt19937 gen;
    boost::random::uniform_int_distribution<> dist;
    boost::mutex mutex;

public:
    multyplyMatrix() : ptr_mtrx{std::make_shared<matrix>()}, now{std::time(0)}, gen{static_cast<std::uint32_t> (now)}, dist{1, 3} {}

    void init_multiply_standart() {


                    {
                        boost::lock_guard<boost::mutex> lock{mutex};
                        auto vec_8 = std::make_unique<std::vector<size_t>>();
                        for (size_t i = 0; i < 8*8; i++)
                            {
                                vec_8->push_back(dist(gen));
                            }
                        boost::timer::cpu_timer tmr;
                        ptr_mtrx->A_8.init_list(*vec_8);
                        ptr_mtrx->B_8.init_list(*vec_8);

                        mult_siml<8, 8>(ptr_mtrx->A_8, ptr_mtrx->B_8, ptr_mtrx->C_8);
                        std::cout <<"Simple\t8x8 " << tmr.format();
                    }
                    {
                        boost::lock_guard<boost::mutex> lock{mutex};
                        auto vec_16 = std::make_unique<std::vector<size_t>>();
                        for (size_t i = 0; i < 16*16; i++)
                            {
                                vec_16->push_back(dist(gen));
                            }
                        boost::timer::cpu_timer tmr;
                        ptr_mtrx->A_16.init_list(*vec_16);
                        ptr_mtrx->B_16.init_list(*vec_16);

                        mult_siml<16, 16>(ptr_mtrx->A_16, ptr_mtrx->B_16, ptr_mtrx->C_16);
                        std::cout <<"Simple\t16x16 " << tmr.format();
                    }
                    {
                        boost::lock_guard<boost::mutex> lock{mutex};
                        auto vec_32 = std::make_unique<std::vector<size_t>>();
                        for (size_t i = 0; i < 32*32; i++)
                            {
                                vec_32->push_back(dist(gen));
                            }
                        boost::timer::cpu_timer tmr;
                        ptr_mtrx->A_32.init_list(*vec_32);
                        ptr_mtrx->B_32.init_list(*vec_32);

                        mult_siml<32, 32>(ptr_mtrx->A_32, ptr_mtrx->B_32, ptr_mtrx->C_32);
                        std::cout <<"Simple\t32x32 " << tmr.format();
                    }
                    {
                        boost::lock_guard<boost::mutex> lock{mutex};
                        auto vec_64 = std::make_unique<std::vector<size_t>>();
                        for (size_t i = 0; i < 64*64; i++)
                            {
                                vec_64->push_back(dist(gen));
                            }
                        boost::timer::cpu_timer tmr;
                        ptr_mtrx->A_64.init_list(*vec_64);
                        ptr_mtrx->B_64.init_list(*vec_64);

                        mult_siml<64, 64>(ptr_mtrx->A_64, ptr_mtrx->B_64, ptr_mtrx->C_64);
                        std::cout <<"Simple\t64x64 " << tmr.format();
                    }

    }
    void init_multiply_Meta() {

                    {
                        boost::lock_guard<boost::mutex> lock{mutex};
                        auto vec_8 = std::make_unique<std::vector<size_t>>();
                        for (size_t i = 0; i < 8*8; i++)
                            {
                                vec_8->push_back(dist(gen));
                            }
                        boost::timer::cpu_timer tmr;
                        ptr_mtrx->A_8.init_list(*vec_8);
                        ptr_mtrx->B_8.init_list(*vec_8);

                        mult_meta<8, 8>(ptr_mtrx->A_8, ptr_mtrx->B_8, ptr_mtrx->C_8);
                        std::cout <<"Metaprog 8x8 " << tmr.format();
                    }
                    {
                        boost::lock_guard<boost::mutex> lock{mutex};
                        auto vec_16 = std::make_unique<std::vector<size_t>>();
                        for (size_t i = 0; i < 16*16; i++)
                            {
                                vec_16->push_back(dist(gen));
                            }
                        boost::timer::cpu_timer tmr;
                        ptr_mtrx->A_16.init_list(*vec_16);
                        ptr_mtrx->B_16.init_list(*vec_16);

                        mult_meta<16, 16>(ptr_mtrx->A_16, ptr_mtrx->B_16, ptr_mtrx->C_16);
                        std::cout <<"Metaprog 16x16 " << tmr.format();
                    }
                    {
                        boost::lock_guard<boost::mutex> lock{mutex};
                        auto vec_32 = std::make_unique<std::vector<size_t>>();
                        for (size_t i = 0; i < 32*32; i++)
                            {
                                vec_32->push_back(dist(gen));
                            }
                        boost::timer::cpu_timer tmr;
                        ptr_mtrx->A_32.init_list(*vec_32);
                        ptr_mtrx->B_32.init_list(*vec_32);

                        mult_meta<32, 32>(ptr_mtrx->A_32, ptr_mtrx->B_32, ptr_mtrx->C_32);
                        std::cout <<"Metaprog 32x32 " << tmr.format();
                    }
                    {
                        boost::lock_guard<boost::mutex> lock{mutex};
                        auto vec_64 = std::make_unique<std::vector<size_t>>();
                        for (size_t i = 0; i < 64*64; i++)
                            {
                                vec_64->push_back(dist(gen));
                            }
                        boost::timer::cpu_timer tmr;
                        ptr_mtrx->A_64.init_list(*vec_64);
                        ptr_mtrx->B_64.init_list(*vec_64);

                        mult_meta<64, 64>(ptr_mtrx->A_64, ptr_mtrx->B_64, ptr_mtrx->C_64);
                        std::cout <<"Metaprog 64x64 " << tmr.format();
                    }
    }
};





