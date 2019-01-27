#include "opencl/opencl.h"
#include "../headers/util.h"

cl_platform_id getPlatform() {
    cl_platform_id platform[1];
    clGetPlatformIDs(1, platform, NULL);
    return *platform;
}

cl_device_id getDevice(cl_platform_id platform) {
    cl_device_id device[1];
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, device, NULL);
    return *device;
}

cl_context createContext(cl_device_id device) {
    void* user_data;
    return clCreateContext(NULL, 1, &device, NULL, user_data, NULL);
}

cl_command_queue createCommandQueue(cl_context context, cl_device_id device) {
    cl_command_queue_properties queue_properties;
    return clCreateCommandQueue(context, device, queue_properties, NULL);
}

cl_mem create2DImage(cl_context context, int size, cl_mem_flags flags, void* host_ptr) {
    cl_image_format image_format;
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_FLOAT;
    cl_image_desc image_desc;
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = size;
    image_desc.image_height = size;
    image_desc.image_row_pitch = 0;
    image_desc.image_slice_pitch = 0;
    image_desc.num_mip_levels = 0;
    image_desc.num_samples = 0;
    image_desc.buffer = NULL;
    cl_int errcode_ret;
    cl_mem result = clCreateImage(context, flags, &image_format, &image_desc, host_ptr, &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        printf(KRED "Error creating image: %d\n", errcode_ret);
    }
    return result;
}

cl_program createProgram(cl_context context, const char *fileName) {
    FILE* file = fopen(fileName, "r");
    if (file == NULL) {
        printf(KRED "Could not open program\n");
        exit(-1);
    }

    fseek(file, 0, SEEK_END);
    int program_size = ftell(file);
    rewind(file);
    char* program_buffer = (char *) calloc (program_size + 1, sizeof(char));
    size_t num_read = fread(program_buffer, sizeof(char), program_size, file);
    if (num_read == 0) {
        printf(KRED "Could not read program\n");
        exit(-1);
    }
    fclose(file);

    cl_int errcode_ret;
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &program_buffer, NULL, &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        printf(KRED "Error creating program: %d\n", errcode_ret);
        exit(-1);
    }
    return program;
}

void buildProgram(cl_program program, cl_device_id device) {
    cl_int r = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (r != CL_SUCCESS) {
        printf(KRED "Error building program: %d\n", (int) r);
        char build_log[16348];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        printf(KYEL "%s\n", build_log);
        exit(-1);
    }
}

cl_kernel createKernel(cl_program program, const char *kernel_name) {
    int errcode_ret;
    cl_kernel kernel = clCreateKernel(program, kernel_name, &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        printf(KRED "Error creating kernel: %d\n", errcode_ret);
        exit(-1);
    }
    return kernel;
}

void setKernelValueArg(cl_kernel kernel, cl_uint arg_index, float arg_value) {
    float value[1] = {arg_value};
    int r = clSetKernelArg(kernel, arg_index, sizeof(float), (const void*) value);
    if (r != CL_SUCCESS) {
        printf(KRED "Error setting argument n %d in kernel: %d\n", arg_index, r);
        exit(-1);
    }
}

void setKernelPointerArg(cl_kernel kernel, cl_uint arg_index, void* arg_value, size_t size) {
    int r = clSetKernelArg(kernel, arg_index, size, arg_value);
    if (r != CL_SUCCESS) {
        printf(KRED "Error setting argument n %d in kernel: %d\n", arg_index, r);
        exit(-1);
    }
}

cl_kernel createTransitionKernel(cl_program program, float a, float b, void* c) {
    cl_kernel kernel = createKernel(program, "transition");
    setKernelValueArg(kernel, 0, a);
    setKernelValueArg(kernel, 1, b);
    setKernelPointerArg(kernel, 2, c, sizeof(cl_mem));
    return kernel;
}

cl_event enqueueNDRangeTransitionKernel(cl_context context, cl_command_queue command_queue, cl_kernel kernel) {
    cl_int eventCreationError;
    cl_event event = clCreateUserEvent(context, &eventCreationError);
    size_t global_work_size[2] = {19, 19};
    cl_int r = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, &event);
    if (r != CL_SUCCESS) {
        printf(KRED "Error equeueing kernel range: %d\n", r);
        exit(-1);
    }
    return event;
}

void finish(cl_command_queue command_queue) {
    int r = clFinish(command_queue);
    if (r != CL_SUCCESS) {
        printf(KRED "Error equeueing kernel range: %d\n", r);
        exit(-1);
    }
}

cl_int getEventExecutionStatus(cl_event event) {
    cl_int status;
    cl_int r = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, (size_t) sizeof(cl_int), &status, NULL);
    if (r != CL_SUCCESS) {
        printf(KRED "Error requesting info from event: %d\n", r);
        exit(-1);
    }
    return status;
}

void* enqueueReadImage(cl_command_queue command_queue, cl_mem image) {
    size_t region[3] = {19, 19, 1};
    size_t image_row_pitch[1] = {0};
    void* ptr = calloc(19 * 19, sizeof(float));
    cl_int r = clEnqueueReadImage(command_queue, image, CL_TRUE, calloc(3, sizeof(size_t)), region, 0, 0, ptr, 0, NULL, NULL);
    if (r != CL_SUCCESS) {
        printf(KRED "Error equeueing map image: %d\n", r);
        exit(-1);
    }
    return ptr;
}

void waitForEvents(cl_uint n, cl_event* e) {
    cl_int r = clWaitForEvents(n, e);
    if (r != CL_SUCCESS) {
        printf(KRED "Error waiting for event: %d\n", r);
        exit(-1);
    }
}

int startCl (void) {
    cl_platform_id platform = getPlatform();
    cl_device_id device = getDevice(platform);

    char param_value[100];
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 100, param_value, NULL);
    printf(KCYN "Plaform version: %s\n", param_value);

    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 100, param_value, NULL);
    printf(KCYN "Vendor: %s\n", param_value);

    cl_context context = createContext(device);
    cl_command_queue command_queue = createCommandQueue(context, device);
    cl_program program = createProgram(context, "transition-kernel.c");
    buildProgram(program, device);
    float* image_ptr = calloc(361, sizeof(float));
    cl_mem image = create2DImage(context, 19, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, (void*) image_ptr);
    cl_kernel kernel = createTransitionKernel(program, 0.5, 0.5, &image);
    cl_event event = enqueueNDRangeTransitionKernel(context, command_queue, kernel);
    waitForEvents(1, &event);
    cl_int executionStatus = getEventExecutionStatus(event);
    switch (executionStatus) {
        case CL_QUEUED:
            printf(KCYN "Kernel queued\n" KWHT);
            break;
        case CL_SUBMITTED:
            printf(KCYN "Kernel submitted\n" KWHT);
            break;
        case CL_RUNNING:
            printf(KCYN "Kernel running\n" KWHT);
            break;
        case CL_COMPLETE:
            printf(KCYN "Kernel completed\n" KWHT);
    }
    image_ptr = (float*) enqueueReadImage(command_queue, image);
    finish(command_queue);
    printf(KCYN "Command queue finished\n" KWHT);
    for (int i = 0; i < 19; i++) {
        for (int j = 0; j < 19; j++) {
            if (image_ptr[i * 19 + j] >= 10) {
                printf("%.0f ", image_ptr[i * 19 + j]);
            } else {
                printf("%.0f  ", image_ptr[i * 19 + j]);
            }
        }
        printf("\n");
    }
    printf(KGRN "\nSUCCESS\n");
    return 0;
}
