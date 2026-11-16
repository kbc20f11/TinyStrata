#include "strata/vision/protocol.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

int main() {
    try {
        int checks = 0;
        const auto check = [&](bool value) { ++checks; if (!value) throw std::runtime_error("Intel vision protocol check failed"); };
        std::string image, output;
        check(strata::vision::parse_enc("ENC plain.png out.sve", image, output));
        check(image == "plain.png" && output == "out.sve");
        check(strata::vision::parse_enc("ENC input with spaces.png plain.sve", image, output));
        check(image == "input with spaces.png" && output == "plain.sve");
        check(strata::vision::parse_enc(u8"ENC\tC:\\画像 フォルダ\\入力.png\tC:\\結果 フォルダ\\出力.sve", image, output));
        check(image == u8"C:\\画像 フォルダ\\入力.png" && output == u8"C:\\結果 フォルダ\\出力.sve");
        const std::vector<std::string> invalid = {
            "", "ENC", "ENC plain.png", "ENC\tin.png", "ENC\t\tout.sve", "ENC\tin.png\t",
            "ENC\tin.png\tout.sve\textra", "ENC\tin\r.png\tout.sve", "ENC\tin\n.png\tout.sve",
            "ENC\tin.png\tout\r.sve", "ENC\tin.png\tout\n.sve", "ENC in.png out.sve\tmore",
            std::string("ENC\tin", 6) + '\0' + ".png\tout.sve", "QUIT\tin.png\tout.sve"
        };
        for (const auto& command : invalid) {
            const auto old_image = image, old_output = output;
            check(!strata::vision::parse_enc(command, image, output));
            check(image == old_image && output == old_output);
        }
        uint32_t width = 7, height = 9;
        check(strata::vision::parse_count("COUNT\t1024\t768", width, height));
        check(width == 1024 && height == 768);
        check(strata::vision::parse_count("COUNT\t1\t16777216", width, height));
        check(width == 1 && height == strata::vision::max_count_dimension);
        const std::vector<std::string> bad_counts = {
            "COUNT", "COUNT 1024 768", "COUNT\t\t768", "COUNT\t1024\t", "COUNT\t0\t768",
            "COUNT\t1024\t0", "COUNT\t-1\t768", "COUNT\t+1\t768", "COUNT\t1.0\t768",
            "COUNT\t1e3\t768", "COUNT\t16777217\t1", "COUNT\t4294967295\t1",
            "COUNT\t999999999999999999999\t1", "COUNT\t32768\t32768", "COUNT\t1024\t768\textra",
            "COUNT\t1024\t768\r", "COUNT\t1024\t768\n", std::string("COUNT\t10\0\t768", 13)
        };
        for (const auto& command : bad_counts) {
            const auto old_width = width, old_height = height;
            check(!strata::vision::parse_count(command, width, height));
            check(width == old_width && height == old_height);
        }
        std::printf("Intel vision protocol PASS checks=%d\n", checks);
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
