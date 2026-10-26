#include "server.h"

hologram_params holo_params_532{WAVELENGTH_532, BEAM_RADIUS_532};
hologram_params holo_params_840{WAVELENGTH_840, BEAM_RADIUS_840};
cv::Mat image_buffer;
char *phase_mask_name;

// rearrange variables
int target_mux[MAX_TWEEZER_NUM] = {0};
int target_muy[MAX_TWEEZER_NUM] = {0};
int dynamic_index[MAX_TWEEZER_NUM] = {0};
int dynamic_num = 0;

volatile int machine_state = MACHINE_IDLE;

static void load_params(char *buffer, hologram_params &params)
{
    if (strtok_s(buffer, ",", &buffer)[0] == 'd')
    {
        params.is_defocus = true;
        params.defocus = atof(strtok_s(buffer, ",", &buffer));
    }
    else
    {
        params.is_defocus = false;
    }
    params.ptnum = atoi(strtok_s(buffer, ",", &buffer));
    for (int i = 0; i < params.ptnum; i++)
    {
        params.mux[i] = atoi(strtok_s(buffer, ",", &buffer));
    }
    for (int i = 0; i < params.ptnum; i++)
    {
        params.muy[i] = atoi(strtok_s(buffer, ",", &buffer));
    }
    for (int i = 0; i < params.ptnum; i++)
    {
        params.amp[i] = atof(strtok_s(buffer, ",", &buffer));
    }
    params.flag = FLAG_NEW;
}

static void mono_slm_checker()
{
    const char *calibration_string;
    if (machine_state == MACHINE_POSITION_CALIBRATING_ARMED)
    {
        calibration_string = "Position";
    }
    else if (machine_state == MACHINE_INTENSITY_CALIBRATING_ARMED)
    {
        calibration_string = "Intensity";
    }
    else
    {
        std::cerr << "Unknown machine state for mono SLM checker" << std::endl;
        return;
    }

    bool calibration_ok = true;
    if (holo_params_532.ptnum < 0 || holo_params_840.ptnum < 0)
    {
        std::cout << calibration_string << " calibration not avaliable with dummy patterns." << '\n';
        calibration_ok = false;
    }
    else
    {
        if (holo_params_532.ptnum == 0)
        {
            if (holo_params_840.ptnum == 0)
            {
                std::cout << calibration_string << " calibration only avaliable with a generated pattern." << '\n';
                calibration_ok = false;
            }
        }
        else if (holo_params_840.ptnum != 0)
        {
            std::cout << calibration_string << " calibration only avaliable with single SLM pattern." << '\n';
            calibration_ok = false;
        }
    }
    if (calibration_ok)
    {
        std::cout << calibration_string << " calibration started." << std::endl;
    }
    else
    {
        machine_state = MACHINE_IDLE;
    }
}

int receive_image(
    int sockfd,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t payload_bytes,
    cv::Mat &image)
{
    constexpr std::uint32_t MAX_WIDTH = 8192;
    constexpr std::uint32_t MAX_HEIGHT = 8192;
    constexpr std::uint64_t MAX_IMAGE_BYTES = 32ull * 1024 * 1024;

    // Validate network-provided values before allocation.
    if (width == 0 || height == 0 ||
        width > MAX_WIDTH || height > MAX_HEIGHT)
    {
        std::cerr << "Invalid dimensions\n";
        return 1;
    }

    const std::uint64_t expected_bytes =
        static_cast<std::uint64_t>(width) * height;

    if (expected_bytes > MAX_IMAGE_BYTES ||
        expected_bytes != payload_bytes)
    {
        std::cerr << "Invalid GRAY8 payload size\n";
        return 1;
    }

    // rows = height, cols = width.
    // CV_8UC1 = 8-bit unsigned, 1-channel.
    image.create(
        static_cast<int>(height),
        static_cast<int>(width),
        CV_16UC1);

    // For a normally allocated Mat, total() * elemSize() is the byte count.
    const std::size_t image_bytes = image.total() * image.elemSize();

    if (image_bytes != expected_bytes)
    {
        image.release();
        std::cerr << "Unexpected OpenCV image allocation size\n";
        return 1;
    }

    if (recv(sockfd, reinterpret_cast<char *>(image.data), image_bytes, 0) != image_bytes)
    {
        image.release(); // Do not leave a partial frame looking valid.
        return 1;
    }

    return 0;
}

int TCP_server()
{
    const int PORT = 56457;
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    const char opt = 1;
    int addrlen = sizeof(address);
    char buffer[BUFFER_SIZE];
    const char *response = "Message received";

    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData))
    {
        printf("WSAStartup failed with error\n");
        return 1;
    }
    // Create socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        std::cerr << "Socket creation failed" << '\n';
        return 1;
    }

    // Forcefully attach socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)))
    {
        std::cerr << "Setsockopt failed: " << WSAGetLastError() << '\n';
        return 1;
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind socket to address
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        std::cerr << "Bind failed" << '\n';
        return 1;
    }

    printf("Server started\n");
    // Listen for incoming connections
    if (listen(server_fd, 3) < 0)
    {
        std::cerr << "Listen failed" << '\n';
        return 1;
    }

    WSAPOLLFD pfd{};
    pfd.fd = server_fd;
    pfd.events = POLLRDNORM;

    while (machine_state != MACHINE_TERMINATING)
    {
        // Accept incoming connection

        const int rc = WSAPoll(&pfd, 1, 200);

        if (rc == 0)
        {
            continue; // Timeout: check running again.
        }

        if (rc == SOCKET_ERROR)
        {
            const int error = WSAGetLastError();
            std::cerr << "WSAPoll failed: " << error << '\n';
            break;
        }

        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (int *)&addrlen)) < 0)
        {
            std::cerr << "Accept failed" << '\n';
            return 1;
        }

        try
        {
            // Receive data from client
            if (machine_state == MACHINE_POSITION_CALIBRATING_ARMED ||
                machine_state == MACHINE_INTENSITY_CALIBRATING_ARMED)
            {
                unsigned int name_size, image_width, image_height, image_size;
                char small_buffer[1024] = {0};
                char *context = nullptr;
                if (recv(new_socket, small_buffer, 4 * sizeof(int), 0) != sizeof(int))
                {
                    std::cerr << "Read error for image name size" << '\n';
                    std::cerr << strerror(errno) << '\n';
                    machine_state = MACHINE_IDLE;
                    continue;
                };
                memcpy(&name_size, small_buffer, sizeof(int));
                memcpy(&image_size, small_buffer + sizeof(int), sizeof(int));
                memcpy(&image_width, small_buffer + 2 * sizeof(int), sizeof(int));
                memcpy(&image_height, small_buffer + 3 * sizeof(int), sizeof(int));
                memset(small_buffer, 0, sizeof(int));
                if (recv(new_socket, small_buffer, name_size, 0) != header_size)
                {
                    std::cerr << "Read error for image name" << '\n';
                    std::cerr << strerror(errno) << '\n';
                    machine_state = MACHINE_IDLE;
                    continue;
                };
                if (receive_image(new_socket, image_width, image_height, image_size, image_buffer))
                {
                    std::cerr << "Read error for image" << '\n';
                    std::cerr << strerror(errno) << '\n';
                    machine_state = MACHINE_IDLE;
                    continue;
                };
                if (machine_state == MACHINE_POSITION_CALIBRATING_ARMED)
                {
                    machine_state = MACHINE_POSITION_CALIBRATING;
                }
                else if (machine_state == MACHINE_INTENSITY_CALIBRATING_ARMED)
                {
                    machine_state = MACHINE_INTENSITY_CALIBRATING;
                }
                continue;
            }
            else if (machine_state == MACHINE_IDLE)
            {
                valread = recv(new_socket, buffer, BUFFER_SIZE, 0);
                if (valread <= 0)
                {
                    std::cerr << "Connection closed or read error" << '\n';
                    std::cerr << strerror(errno) << '\n';
                }

                std::cout << "Received message: " << buffer << '\n';

                // Check if message contains "go"
                if (strstr(buffer, "go") != nullptr)
                {
                    // Send response back to client
                    send(new_socket, response, strlen(response), 0);
                    std::cout << "Sent response: " << response << '\n';
                    // The sent in string should be an array of numbers separated by "," with a char 'g' at the end.
                    // The numbers should be a concanation of static_num, dynamic_num, static_freq, dynamic_freq, and dynamic_list;
                    // the latter three are only needed if dynamic_num > 0
                    holo_params_532.ptnum = 0;
                    holo_params_840.ptnum = 0;
                    char *rest = nullptr;
                    char *first = strtok_s(buffer, ",", &rest);
                    if (first == nullptr)
                    {
                        std::cout << "No command received." << '\n';
                        machine_state = MACHINE_IDLE;
                        continue;
                    }
                    if (first[0] == '5')
                    {
                        // if receiving 532 commands first
                        char *ender = nullptr;
                        if constexpr (SLM_type != SLMType::SLM_840)
                        {
                            // if we have a 532 SLM, then load 532 commands

                            load_params(rest, holo_params_532);

                            ender = strtok_s(rest, ",", &rest);
                            if (ender == nullptr)
                            {
                                std::cout << "Error in receiving." << '\n';
                                machine_state = MACHINE_IDLE;
                                continue;
                            }
                            if (ender[0] == '8')
                            {
                                // then after loading 532 commands, if we also receive 840 commands
                                if constexpr (SLM_type != SLMType::SLM_532)
                                {
                                    // and if we have a 840 SLM, then load 840 data
                                    load_params(rest, holo_params_840);
                                    if (strtok_s(rest, ",", &rest)[0] != 'g')
                                    {
                                        // check if the command ends with 'go'
                                        std::cout << "Error in receiving." << '\n';
                                        machine_state = MACHINE_IDLE;
                                    }
                                    else
                                    {
                                        machine_state = MACHINE_GENERATING;
                                    }
                                }
                                else
                                {
                                    // if we do not have a 840 SLM, then just ignore the 840 commands and check if the commands end with 'go'
                                    while (char *temp = strtok_s(rest, ",", &rest))
                                    {
                                        if (temp[0] == 'g')
                                        {
                                            machine_state = MACHINE_GENERATING;
                                            break;
                                        }
                                    }
                                }
                            }
                            else if (ender[0] != 'g')
                            {
                                // if no 840 commands just check if the command ends with 'go'
                                std::cout << "Error in receiving." << '\n';
                                machine_state = MACHINE_IDLE;
                            }
                            else
                            {
                                machine_state = MACHINE_GENERATING;
                            }
                        }
                        else
                        {
                            // if we only have 840 SLM, then try loading 840 commands
                            while (char *temp = strtok_s(rest, ",", &rest))
                            {
                                if (temp[0] == '8')
                                {
                                    load_params(rest, holo_params_840);
                                    if (strtok_s(rest, ",", &rest)[0] != 'g')
                                    {
                                        // check if the command ends with 'go'
                                        std::cout << "Error in receiving." << '\n';
                                        machine_state = MACHINE_IDLE;
                                    }
                                    else
                                    {
                                        machine_state = MACHINE_GENERATING;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    else if (first[0] == '8')
                    {
                        // if receiving 840 commands first
                        char *ender = nullptr;
                        if constexpr (SLM_type != SLMType::SLM_532)
                        {
                            // if we have a 840 SLM, then load 840 commands
                            load_params(rest, holo_params_840);
                            ender = strtok_s(rest, ",", &rest);
                            if (ender == nullptr)
                            {
                                std::cout << "Error in receiving." << '\n';
                                machine_state = MACHINE_IDLE;
                                continue;
                            }
                            if (ender[0] == '5')
                            {
                                // then after loading 840 commands, if we also receive 532 commands
                                if constexpr (SLM_type != SLMType::SLM_840)
                                {
                                    // and if we have a 532 SLM, then load 532 data
                                    load_params(rest, holo_params_532);
                                    if (strtok_s(rest, ",", &rest)[0] != 'g')
                                    {
                                        // check if the command ends with 'go'
                                        std::cout << "Error in receiving." << '\n';
                                        machine_state = MACHINE_IDLE;
                                    }
                                    else
                                    {
                                        machine_state = MACHINE_GENERATING;
                                    }
                                }
                                else
                                {
                                    // if we do not have a 532 SLM, then just ignore the 532 commands and check if the commands end with 'go'
                                    while (char *temp = strtok_s(rest, ",", &rest))
                                    {
                                        if (temp[0] == 'g')
                                        {
                                            machine_state = MACHINE_GENERATING;
                                            break;
                                        }
                                    }
                                }
                            }
                            else if (ender[0] != 'g')
                            {
                                // if no 532 commands just check if the command ends with 'go'
                                std::cout << "Error in receiving." << '\n';
                                machine_state = MACHINE_IDLE;
                            }
                            else
                            {
                                machine_state = MACHINE_GENERATING;
                            }
                        }
                        else
                        {
                            // if we only have 532 SLM, then try loading 532 commands
                            while (char *temp = strtok_s(rest, ",", &rest))
                            {
                                if (temp[0] == '5')
                                {
                                    load_params(rest, holo_params_532);
                                    if (strtok_s(rest, ",", &rest)[0] != 'g')
                                    {
                                        // check if the command ends with 'go'
                                        std::cout << "Error in receiving." << '\n';
                                        machine_state = MACHINE_IDLE;
                                    }
                                    else
                                    {
                                        machine_state = MACHINE_GENERATING;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    else
                    {
                        std::cout << "Error in receiving." << '\n';
                        machine_state = MACHINE_IDLE;
                    }
                }
                else if (strstr(buffer, "intensity_calibration") != nullptr)
                {
                    machine_state = MACHINE_INTENSITY_CALIBRATING_ARMED;
                    mono_slm_checker();
                }
                else if (strstr(buffer, "position_calibration") != nullptr)
                {
                    machine_state = MACHINE_POSITION_CALIBRATING_ARMED;
                    mono_slm_checker();
                }
                else if (strstr(buffer, "ud") != nullptr)
                {
                    // TODO: rearrange code

                }
                else if (strstr(buffer, "dummy") != nullptr)
                {
                    // let the program generate a dummy array
                    machine_state = MACHINE_DUMMY;
                    std::cout << "Generate a dummy array" << '\n';
                }
                else if (strstr(buffer, "zero") != nullptr)
                {
                    // let the program generate a zero array
                    std::cout << "Generate no phase pattern" << '\n';
                    machine_state = MACHINE_ZERO;
                }
                else if (strstr(buffer, "load_map") != nullptr)
                {
                    // let the program load a map
                    machine_state = MACHINE_IDLE;
                }
                else if (strstr(buffer, "save_phase") != nullptr)
                {
                    phase_mask_name = strtok(buffer, ",");
                    machine_state = MACHINE_SAVE_HOLOGRAM;
                }
                else if (strstr(buffer, "load_phase") != nullptr)
                {
                    phase_mask_name = strtok(buffer, ",");
                    machine_state = MACHINE_LOAD_HOLOGRAM;
                }
                else if (strstr(buffer, "stop") != nullptr)
                {
                    // stop the program
                    machine_state = MACHINE_TERMINATING;
                    break;
                }
            }
        }
        catch (std::exception &e)
        {
            std::cerr << "Error during TCP: " << e.what() << '\n';
        }

        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))
            break;
        // Clear buffer
        memset(buffer, 0, BUFFER_SIZE);
    }
    memset(buffer, 0, BUFFER_SIZE);
    // Close sockets
    closesocket(new_socket);
    closesocket(server_fd);
    printf("Server thread exits successfully\n");

    return 0;
}
