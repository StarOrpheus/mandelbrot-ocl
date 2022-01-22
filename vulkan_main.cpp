#include "vulkan/vulkan.hpp"
#include "const.h"

#include <iostream>
#include <fstream>
#include <sstream>

constexpr size_t M = 3840;
constexpr size_t N = 2160;

constexpr auto AppName    = "Mandelbrot Renderer";
constexpr auto EngineName = "Vulkan.hpp";

static inline
std::string load_source_file(char const* file_name)
{
    std::ifstream input(file_name);
    input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    std::stringstream ss;
    ss << input.rdbuf();
    return ss.str();
}

int main( int /*argc*/, char ** /*argv*/ )
{
    constexpr auto FrameSize = N * M;
    constexpr auto BufferSize = FrameSize * sizeof(precise_t);

    try {
        vk::ApplicationInfo applicationInfo( AppName, 1, EngineName, 1, VK_API_VERSION_1_1 );

        vk::InstanceCreateInfo instanceCreateInfo( {}, &applicationInfo );

        vk::Instance instance = vk::createInstance( instanceCreateInfo );

        auto physicalDevice
            = instance.enumeratePhysicalDevices().front();

        std::vector<vk::QueueFamilyProperties> queueFamilyProperties
            = physicalDevice.getQueueFamilyProperties();

        std::cout << "Selected device "
                  << physicalDevice.getProperties().deviceName << std::endl;

        auto qfpIter
            = std::find_if(queueFamilyProperties.begin(),
                           queueFamilyProperties.end(),
                           [] (vk::QueueFamilyProperties const& P) {
            return P.queueFlags & vk::QueueFlagBits::eCompute;
        });
        if (qfpIter == queueFamilyProperties.end())
            throw std::runtime_error("Satysfying queueFamilyProperties not found");
        uint32_t const computeQueueFamilyIndex
            = std::distance(queueFamilyProperties.begin(), qfpIter);

        float queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo(
                vk::DeviceQueueCreateFlags(), computeQueueFamilyIndex, 1, &queuePriority);
        vk::DeviceCreateInfo DeviceCreateInfo(vk::DeviceCreateFlags(), deviceQueueCreateInfo);
        vk::Device device
            = physicalDevice.createDevice(vk::DeviceCreateInfo(vk::DeviceCreateFlags(), deviceQueueCreateInfo));

        vk::BufferCreateInfo BufferCreateInfo{
            vk::BufferCreateFlags(),
            FrameSize * sizeof(precise_t),
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::SharingMode::eExclusive,
            1,
            &computeQueueFamilyIndex
        };

        vk::Buffer OutBuffer = device.createBuffer(BufferCreateInfo);

        vk::MemoryRequirements OutBufferMemoryRequirements
            = device.getBufferMemoryRequirements(OutBuffer);

        vk::PhysicalDeviceMemoryProperties MemoryProperties
            = physicalDevice.getMemoryProperties();

        uint32_t MemoryTypeIndex = uint32_t(~0);
        vk::DeviceSize MemoryHeapSize = uint32_t(~0);
        for (uint32_t CurrentMemoryTypeIndex = 0;
             CurrentMemoryTypeIndex < MemoryProperties.memoryTypeCount;
             ++CurrentMemoryTypeIndex)
        {
            vk::MemoryType MemoryType
                = MemoryProperties.memoryTypes[CurrentMemoryTypeIndex];
            if ((vk::MemoryPropertyFlagBits::eHostVisible & MemoryType.propertyFlags) &&
                (vk::MemoryPropertyFlagBits::eHostCoherent & MemoryType.propertyFlags))
            {
                MemoryHeapSize = MemoryProperties.memoryHeaps[MemoryType.heapIndex].size;
                MemoryTypeIndex = CurrentMemoryTypeIndex;
                break;
            }
        }

        std::cout << "Memory Type Index: " << MemoryTypeIndex << std::endl;
        std::cout << "Memory Heap Size : " << MemoryHeapSize / 1024 / 1024 / 1024 << " GB" << std::endl;

        vk::MemoryAllocateInfo OutBufferMemoryAllocateInfo(
                OutBufferMemoryRequirements.size, MemoryTypeIndex);
        vk::DeviceMemory OutBufferMemory = device.allocateMemory(
                OutBufferMemoryAllocateInfo);

        device.bindBufferMemory(OutBuffer, OutBufferMemory, 0);

        auto ShaderContents = load_source_file("mandelbrot.spv");

        device.destroy();
        instance.destroy();
    } catch ( vk::SystemError & err ) {
        std::cout << "vk::SystemError: " << err.what() << std::endl;
        exit( -1 );
    } catch ( std::exception & err ) {
        std::cout << "std::exception: " << err.what() << std::endl;
        exit( -1 );
    } catch ( ... ) {
        std::cout << "unknown error\n";
        exit( -1 );
    }
}
