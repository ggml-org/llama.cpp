#include "common.h"
#include "mimi-model.h"

#include <vector>
#include <fstream>


/**
 * This file is used for testing and showcase how to use "mimi_model" class.
 * Please keep it simple and easy to understand.
 */

int main(int argc, const char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf codes.txt [output.wav]\n", argv[0]);
        fprintf(stderr, "  Format of codes.txt file: one code per line\n");
        fprintf(stderr, "  Replace codes.txt with dummy0 and dummy1 for testing\n");
        fprintf(stderr, "    dummy0: using code 1, 2, 3,..., 96, used for logits matching\n");
        fprintf(stderr, "    dummy1: using code that will outputs 'wah hello there' sound\n");
        return 1;
    }

    const char * model_path = argv[1];
    const char * codes_path = argv[2];
    const char * out_path   = argc < 4 ? "output.wav" : argv[3];

    // load codes
    std::vector<int> codes;
    if (strcmp(codes_path, "dummy0") == 0) {
        printf("Using dummy0 codes\n");
        codes.resize(32 * 3); // [n_codes_per_embd = 32, n_codes = 3]
        for (int i = 0; i < (int)codes.size(); i++) {
            codes[i] = i;
        }
    } else if (strcmp(codes_path, "dummy1") == 0) {
        printf("Using dummy1 codes\n");
        codes = {
            1049 ,1415 ,1962 ,914  ,1372 ,704  ,1922 ,2036 ,288  ,968  ,193  ,1139 ,897  ,897  ,1243 ,1511 ,
            1597 ,175  ,1280 ,1202 ,1911 ,85   ,47   ,692  ,632  ,251  ,1553 ,1735 ,1577 ,132  ,471  ,433  ,
            1325 ,1539 ,1943 ,1601 ,141  ,257  ,564  ,1435 ,876  ,1096 ,636  ,61   ,1497 ,1010 ,485  ,284  ,
            839  ,776  ,878  ,1719 ,1069 ,1302 ,893  ,2005 ,875  ,908  ,586  ,2001 ,186  ,1932 ,1765 ,721  ,
            592  ,1046 ,1588 ,1670 ,1485 ,1141 ,34   ,1465 ,1156 ,1938 ,435  ,753  ,1418 ,277  ,391  ,1741 ,
            1440 ,117  ,723  ,412  ,642  ,1717 ,131  ,37   ,345  ,112  ,1979 ,2034 ,1822 ,1536 ,1281 ,56   ,
            1341 ,803  ,568  ,568  ,1370 ,1995 ,1063 ,892  ,273  ,895  ,1226 ,354  ,1726 ,1541 ,1607 ,615  ,
            985  ,1499 ,1736 ,1838 ,702  ,1345 ,1657 ,511  ,1774 ,1787 ,945  ,1927 ,947  ,952  ,1418 ,916  ,
            1239 ,1457 ,1021 ,341  ,284  ,882  ,474  ,1559 ,1923 ,273  ,1330 ,1406 ,1782 ,19   ,116  ,887  ,
            1146 ,1307 ,983  ,1237 ,1407 ,1350 ,1960 ,1255 ,878  ,1979 ,1500 ,1939 ,1415 ,88   ,1702 ,1253 ,
            1778 ,2    ,10   ,1279 ,999  ,1549 ,1049 ,373  ,1355 ,1200 ,1466 ,1009 ,75   ,2042 ,1725 ,916  ,
            1636 ,1135 ,833  ,830  ,1758 ,2015 ,1275 ,1675 ,287  ,744  ,89   ,430  ,1724 ,1232 ,1692 ,535  ,
            1485 ,1287 ,973  ,1815 ,314  ,2020 ,424  ,1085 ,982  ,1994 ,1563 ,1269 ,1769 ,1681 ,1082 ,1666 ,
            1622 ,1039 ,1209 ,32   ,679  ,732  ,976  ,1462 ,805  ,402  ,1150 ,170  ,1529 ,2013 ,350  ,1175 ,
            757  ,1124 ,1091 ,1369 ,1061 ,415  ,1217 ,1135 ,1360 ,1578 ,1205 ,1785 ,1835 ,1241 ,14   ,716  ,
            480  ,716  ,681  ,1686 ,1624 ,335  ,865  ,1356 ,1688 ,307  ,366  ,541  ,1262 ,1167 ,59   ,269  ,
            1899 ,1798 ,1606 ,1307 ,1549 ,1814 ,114  ,483  ,958  ,1919 ,1179 ,898  ,834  ,1526 ,386  ,447  ,
            1481 ,201  ,779  ,419  ,430  ,1451 ,1000 ,156  ,1062 ,615  ,1353 ,414  ,1214 ,1487 ,882  ,32   ,
            840  ,1517 ,334  ,1143 ,823  ,454  ,725  ,1298 ,1325 ,649  ,1737 ,913  ,685  ,761  ,2010 ,63   ,
            1397 ,1299 ,765  ,1158 ,1809 ,1299 ,1585 ,1776 ,625  ,1539 ,830  ,1563 ,461  ,308  ,1438 ,321  ,
            82   ,886  ,1836 ,325  ,1976 ,761  ,359  ,1136 ,1720 ,2036 ,904  ,719  ,526  ,1567 ,145  ,1860 ,
            1565 ,1786 ,1400 ,1696 ,232  ,1736 ,512  ,518  ,1895 ,1854 ,1584 ,1393 ,1869 ,1702 ,789  ,1986 ,
            116  ,521  ,150  ,1597 ,727  ,1916 ,815  ,1826 ,1382 ,653  ,1596 ,286  ,1373 ,177  ,1397 ,1009 ,
            1449 ,353  ,877  ,93   ,266  ,1853 ,1255 ,872  ,1974 ,556  ,1885 ,857  ,992  ,5    ,1921 ,1849 ,
            1038 ,1912 ,464  ,795  ,747  ,56   ,124  ,431  ,1868 ,609  ,855  ,1522 ,912  ,1709 ,1507 ,1062 ,
            1015 ,1357 ,1487 ,4    ,253  ,1871 ,933  ,215  ,1228 ,633  ,1306 ,2024 ,1453 ,900  ,457  ,471  ,
            436  ,1311 ,870  ,1032 ,134  ,984  ,1983 ,1103 ,1627 ,1627 ,414  ,1845 ,583  ,1699 ,1458 ,2018 ,
            150  ,450  ,1114 ,369  ,267  ,1273 ,1136 ,1578 ,1063 ,1820 ,120  ,779  ,652  ,1266 ,1929 ,1213 ,
            159  ,297  ,1703 ,819  ,93   ,247  ,1366 ,144  ,1617 ,1428 ,812  ,121  ,1637 ,1620 ,289  ,1557 ,
            1414 ,971  ,476  ,1685 ,428  ,1802 ,653  ,1290 ,614  ,1663 ,1528 ,1344 ,798  ,1027 ,1305 ,990  ,
            1740 ,1154 ,1839 ,912  ,731  ,602  ,1064 ,1508 ,834  ,1387 ,252  ,745  ,1034 ,1102 ,965  ,696  ,
            1971 ,1729 ,666  ,282  ,1993 ,1551 ,1703 ,1124 ,1628 ,1725 ,107  ,808  ,1096 ,1753 ,500  ,677  ,
        };
    } else {
        std::ifstream fin(codes_path);
        if (!fin) {
            fprintf(stderr, "Error: cannot open codes file: %s\n", codes_path);
            return 1;
        }
        std::string line;
        while (std::getline(fin, line)) {
            // Skip empty lines
            if (line.empty()) continue;
            try {
                int code = std::stoi(line);
                codes.push_back(code);
            } catch (const std::exception& e) {
                fprintf(stderr, "Error parsing code: %s\n", line.c_str());
                return 1;
            }
        }
        if (codes.empty()) {
            fprintf(stderr, "Error: no codes found in file: %s\n", codes_path);
            return 1;
        }

        printf("Loaded %d codes from %s\n", (int)codes.size(), codes_path);
    }

    mimi_model model(model_path, true);
    std::vector<float> wav_data = model.decode(codes);

    // print first 20 values
    printf("Number of output samples: %d\n", (int)wav_data.size());
    printf("First 20 samples:\n");
    for (int i = 0; i < 20; i++) {
        printf("%2.4f, ", wav_data[i]);
    }
    printf("...\n");

    // write to wav
    printf("Writing to %s\n", out_path);
    save_wav16(out_path, wav_data, model.get_sample_rate());
}
