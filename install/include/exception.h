
#include <iostream>
#include <sstream>

namespace ct
{

/**
 * @brief custom exception to be thrown when there is any
 * runtime error
 */
class CudaRTException : public std::exception
{
public:
    /**
     * @brief Construct a new CudaRTException object
     * and constructs the message with file, line and functiona name information
     * @param status cuda error
     * @param func_str function info
     * @param file file name
     * @param line line number
     */
    CudaRTException(
        cudaError_t status,
        const char* const func_str,
        const char* const file,
        const int line)
    {
        std::stringstream stream;
        stream << "CUDA Runtime Error at: " << file << ":" << line << "\n";
        stream << "Function: " << func_str << "\n";
        stream << cudaGetErrorString(status) << "\n";
        m_message = stream.str();
    }

    /**
     * @brief Construct a new CudaRTException object
     * with a given message
     * @param message message to show
     */
    CudaRTException(const std::string& message) : m_message(message) {}

    /**
     * @brief get error details
     *
     * @return const char* to error message
     */
    const char* what() const noexcept override
    {
        return m_message.c_str();
    }

private:
    std::string m_message;
};

}  // namespace ct