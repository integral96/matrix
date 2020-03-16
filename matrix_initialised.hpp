#pragma once


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
    my_matrix<128, 128> A_128;
    my_matrix<256, 256> A_256;
    ///////
    my_matrix<8, 8>     B_8;
    my_matrix<16, 16>   B_16;
    my_matrix<32, 32>   B_32;
    my_matrix<64, 64>   B_64;
    my_matrix<128, 128> B_128;
    my_matrix<256, 256> B_256;
    ///////
    my_matrix<8, 8>     C_8;
    my_matrix<16, 16>   C_16;
    my_matrix<32, 32>   C_32;
    my_matrix<64, 64>   C_64;
    my_matrix<128, 128> C_128;
    my_matrix<256, 256> C_256;
};


boost::mutex mutex;

void init_multiply_matrix() {
    matrix mtrx;
    std::time_t now = std::time(0);
    boost::random::mt19937 gen{static_cast<std::uint32_t> (now)};
    boost::random::uniform_int_distribution<> dist{1, 3};

    std::vector<boost::thread> thrd;
    for(int j = 0; j < 11; ++j) {
        
        thrd.push_back(boost::thread([j, &mtrx, &gen, &dist]() mutable {
            
            if(j == 0) {
                    boost::lock_guard<boost::mutex> lock{mutex};
                    auto vec_8 = std::make_unique<std::vector<size_t>>();
                    for (size_t i = 0; i < 8*8; i++)
                        {
                            vec_8->push_back(dist(gen));
                        }
                    boost::timer::cpu_timer tmr;
                    mtrx.A_8.init_list(*vec_8);
                    mtrx.B_8.init_list(*vec_8);

                    mult_siml<8, 8>(mtrx.A_8, mtrx.B_8, mtrx.C_8);
                    std::cout <<"Simple\t8x8 " << tmr.format();
            }
            if(j == 1) {
                    boost::lock_guard<boost::mutex> lock{mutex};
                    auto vec_16 = std::make_unique<std::vector<size_t>>();
                    for (size_t i = 0; i < 16*16; i++)
                        {
                            vec_16->push_back(dist(gen));
                        }
                    boost::timer::cpu_timer tmr;
                    mtrx.A_16.init_list(*vec_16);
                    mtrx.B_16.init_list(*vec_16);

                    mult_siml<16, 16>(mtrx.A_16, mtrx.B_16, mtrx.C_16);
                    std::cout <<"Simple\t16x16 " << tmr.format();
            }
            if(j == 2) {
                    boost::lock_guard<boost::mutex> lock{mutex};
                    auto vec_32 = std::make_unique<std::vector<size_t>>();
                    for (size_t i = 0; i < 32*32; i++)
                        {
                            vec_32->push_back(dist(gen));
                        }
                    boost::timer::cpu_timer tmr;
                    mtrx.A_32.init_list(*vec_32);
                    mtrx.B_32.init_list(*vec_32);

                    mult_siml<32, 32>(mtrx.A_32, mtrx.B_32, mtrx.C_32);
                    std::cout <<"Simple\t32x32 " << tmr.format();
            }
            if(j == 3) {
                    boost::lock_guard<boost::mutex> lock{mutex};
                    auto vec_64 = std::make_unique<std::vector<size_t>>();
                    for (size_t i = 0; i < 64*64; i++)
                        {
                            vec_64->push_back(dist(gen));
                        }
                    boost::timer::cpu_timer tmr;
                    mtrx.A_64.init_list(*vec_64);
                    mtrx.B_64.init_list(*vec_64);

                    mult_siml<64, 64>(mtrx.A_64, mtrx.B_64, mtrx.C_64);
                    std::cout <<"Simple\t64x64 " << tmr.format();
            }
            if(j == 4) {
                    boost::lock_guard<boost::mutex> lock{mutex};
                    auto vec_128 = std::make_unique<std::vector<size_t>>();
                    for (size_t i = 0; i < 128*128; i++)
                        {
                            vec_128->push_back(dist(gen));
                        }
                    boost::timer::cpu_timer tmr;
                    mtrx.A_128.init_list(*vec_128);
                    mtrx.B_128.init_list(*vec_128);

                    mult_siml<128, 128>(mtrx.A_128, mtrx.B_128, mtrx.C_128);
                    std::cout <<"Simple\t128x128 " << tmr.format();
            }
            if(j == 5) {
                    boost::lock_guard<boost::mutex> lock{mutex};
                    auto vec_256 = std::make_unique<std::vector<size_t>>();
                    for (size_t i = 0; i < 256*256; i++)
                        {
                            vec_256->push_back(dist(gen));
                        }
                    boost::timer::cpu_timer tmr;
                    mtrx.A_256.init_list(*vec_256);
                    mtrx.B_256.init_list(*vec_256);

                    mult_siml<256, 256>(mtrx.A_256, mtrx.B_256, mtrx.C_256);
                    std::cout <<"Simple\t256x256 " << tmr.format();
            }
            /////////META//////////////////////////////////////////
            if(j == 6) {
                    boost::lock_guard<boost::mutex> lock{mutex};
                    auto vec_8 = std::make_unique<std::vector<size_t>>();
                    for (size_t i = 0; i < 8*8; i++)
                        {
                            vec_8->push_back(dist(gen));
                        }
                    boost::timer::cpu_timer tmr;
                    mtrx.A_8.init_list(*vec_8);
                    mtrx.B_8.init_list(*vec_8);

                    mult_meta<8, 8>(mtrx.A_8, mtrx.B_8, mtrx.C_8);
                    std::cout <<"Metaprog 8x8 " << tmr.format();
            }
            if(j == 7) {
                    boost::lock_guard<boost::mutex> lock{mutex};
                    auto vec_16 = std::make_unique<std::vector<size_t>>();
                    for (size_t i = 0; i < 16*16; i++)
                        {
                            vec_16->push_back(dist(gen));
                        }
                    boost::timer::cpu_timer tmr;
                    mtrx.A_16.init_list(*vec_16);
                    mtrx.B_16.init_list(*vec_16);

                    mult_meta<16, 16>(mtrx.A_16, mtrx.B_16, mtrx.C_16);
                    std::cout <<"Metaprog 16x16 " << tmr.format();
            }
            if(j == 8) {
                    boost::lock_guard<boost::mutex> lock{mutex};
                    auto vec_32 = std::make_unique<std::vector<size_t>>();
                    for (size_t i = 0; i < 32*32; i++)
                        {
                            vec_32->push_back(dist(gen));
                        }
                    boost::timer::cpu_timer tmr;
                    mtrx.A_32.init_list(*vec_32);
                    mtrx.B_32.init_list(*vec_32);

                    mult_meta<32, 32>(mtrx.A_32, mtrx.B_32, mtrx.C_32);
                    std::cout <<"Metaprog 32x32 " << tmr.format();
            }
            if(j == 9) {
                    boost::lock_guard<boost::mutex> lock{mutex};
                    auto vec_64 = std::make_unique<std::vector<size_t>>();
                    for (size_t i = 0; i < 64*64; i++)
                        {
                            vec_64->push_back(dist(gen));
                        }
                    boost::timer::cpu_timer tmr;
                    mtrx.A_64.init_list(*vec_64);
                    mtrx.B_64.init_list(*vec_64);

                    mult_meta<64, 64>(mtrx.A_64, mtrx.B_64, mtrx.C_64);
                    std::cout <<"Metaprog 64x64 " << tmr.format();
            }
            if(j == 10) {
                    boost::lock_guard<boost::mutex> lock{mutex};
                    auto vec_128 = std::make_unique<std::vector<size_t>>();
                    for (size_t i = 0; i < 128*128; i++)
                        {
                            vec_128->push_back(dist(gen));
                        }
                    boost::timer::cpu_timer tmr;
                    mtrx.A_128.init_list(*vec_128);
                    mtrx.B_128.init_list(*vec_128);

                    mult_meta<128, 128>(mtrx.A_128, mtrx.B_128, mtrx.C_128);
                    std::cout <<"Metaprog 128x128 " << tmr.format();
            }
            
        }));
    }
    for(auto& x : thrd) x.join();
}


