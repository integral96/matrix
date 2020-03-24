#include <matrix_initialised.hpp>


#include <boost/lexical_cast.hpp>

boost::mutex mutex;

void time_multiple(boost::promise<std::string>& pr, multyplyMatrix& m_mtrx) {
    std::string id = boost::lexical_cast<std::string>(boost::this_thread::get_id());
    m_mtrx.init_multiply_Meta();
    std::cout << "Print thread " << id << " starting.\n";
    boost::this_thread::sleep_for(boost::chrono::seconds(15));
    pr.set_value(id);
}
void thrFunc(boost::shared_future<std::string> sf, int waitfor) {
    while (sf.wait_for(boost::chrono::seconds(waitfor)) == boost::future_status::timeout) {
        boost::lock_guard<boost::mutex> lg(mutex);
        std::cout << "Subscriber thread " << boost::this_thread::get_id() << " waiting...\n";
    }
    std::cout << "\nSubscriber thread " << boost::this_thread::get_id() << " got " << sf.get() << std::endl;
}
int main() {
    multyplyMatrix mtrx;
    boost::promise<std::string> prom;
    boost::future<std::string> futur(prom.get_future());
    boost::shared_future<std::string> shfut(std::move(futur));

    boost::thread print_mtrx(boost::bind(time_multiple, boost::ref(prom), boost::ref(mtrx)));

    boost::thread subscribe1(boost::bind(thrFunc, shfut, 2));
    boost::thread subscribe2(boost::bind(thrFunc, shfut, 4));
    boost::thread subscribe3(boost::bind(thrFunc, shfut, 6));

    print_mtrx.join();
    subscribe1.join();
    subscribe2.join();
    subscribe3.join();


	return 0;
}

