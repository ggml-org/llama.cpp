#include "common.h"
#include "mimi-model.h"

#include <vector>
#include <fstream>
#include <string.h> // strcmp


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
            1049 ,1597 ,1325 ,839  ,592  ,1440 ,1341 ,985  ,1239 ,1146 ,1778 ,1636 ,1485 ,1622 ,757  ,480  ,
            1899 ,1481 ,840  ,1397 ,82   ,1565 ,116  ,1449 ,1038 ,1015 ,436  ,150  ,159  ,1414 ,1740 ,1971 ,
            1415 ,175  ,1539 ,776  ,1046 ,117  ,803  ,1499 ,1457 ,1307 ,2    ,1135 ,1287 ,1039 ,1124 ,716  ,
            1798 ,201  ,1517 ,1299 ,886  ,1786 ,521  ,353  ,1912 ,1357 ,1311 ,450  ,297  ,971  ,1154 ,1729 ,
            1962 ,1280 ,1943 ,878  ,1588 ,723  ,568  ,1736 ,1021 ,983  ,10   ,833  ,973  ,1209 ,1091 ,681  ,
            1606 ,779  ,334  ,765  ,1836 ,1400 ,150  ,877  ,464  ,1487 ,870  ,1114 ,1703 ,476  ,1839 ,666  ,
            914  ,1202 ,1601 ,1719 ,1670 ,412  ,568  ,1838 ,341  ,1237 ,1279 ,830  ,1815 ,32   ,1369 ,1686 ,
            1307 ,419  ,1143 ,1158 ,325  ,1696 ,1597 ,93   ,795  ,4    ,1032 ,369  ,819  ,1685 ,912  ,282  ,
            1372 ,1911 ,141  ,1069 ,1485 ,642  ,1370 ,702  ,284  ,1407 ,999  ,1758 ,314  ,679  ,1061 ,1624 ,
            1549 ,430  ,823  ,1809 ,1976 ,232  ,727  ,266  ,747  ,253  ,134  ,267  ,93   ,428  ,731  ,1993 ,
            704  ,85   ,257  ,1302 ,1141 ,1717 ,1995 ,1345 ,882  ,1350 ,1549 ,2015 ,2020 ,732  ,415  ,335  ,
            1814 ,1451 ,454  ,1299 ,761  ,1736 ,1916 ,1853 ,56   ,1871 ,984  ,1273 ,247  ,1802 ,602  ,1551 ,
            1922 ,47   ,564  ,893  ,34   ,131  ,1063 ,1657 ,474  ,1960 ,1049 ,1275 ,424  ,976  ,1217 ,865  ,
            114  ,1000 ,725  ,1585 ,359  ,512  ,815  ,1255 ,124  ,933  ,1983 ,1136 ,1366 ,653  ,1064 ,1703 ,
            2036 ,692  ,1435 ,2005 ,1465 ,37   ,892  ,511  ,1559 ,1255 ,373  ,1675 ,1085 ,1462 ,1135 ,1356 ,
            483  ,156  ,1298 ,1776 ,1136 ,518  ,1826 ,872  ,431  ,215  ,1103 ,1578 ,144  ,1290 ,1508 ,1124 ,
            288  ,632  ,876  ,875  ,1156 ,345  ,273  ,1774 ,1923 ,878  ,1355 ,287  ,982  ,805  ,1360 ,1688 ,
            958  ,1062 ,1325 ,625  ,1720 ,1895 ,1382 ,1974 ,1868 ,1228 ,1627 ,1063 ,1617 ,614  ,834  ,1628 ,
            968  ,251  ,1096 ,908  ,1938 ,112  ,895  ,1787 ,273  ,1979 ,1200 ,744  ,1994 ,402  ,1578 ,307  ,
            1919 ,615  ,649  ,1539 ,2036 ,1854 ,653  ,556  ,609  ,633  ,1627 ,1820 ,1428 ,1663 ,1387 ,1725 ,
            193  ,1553 ,636  ,586  ,435  ,1979 ,1226 ,945  ,1330 ,1500 ,1466 ,89   ,1563 ,1150 ,1205 ,366  ,
            1179 ,1353 ,1737 ,830  ,904  ,1584 ,1596 ,1885 ,855  ,1306 ,414  ,120  ,812  ,1528 ,252  ,107  ,
            1139 ,1735 ,61   ,2001 ,753  ,2034 ,354  ,1927 ,1406 ,1939 ,1009 ,430  ,1269 ,170  ,1785 ,541  ,
            898  ,414  ,913  ,1563 ,719  ,1393 ,286  ,857  ,1522 ,2024 ,1845 ,779  ,121  ,1344 ,745  ,808  ,
            897  ,1577 ,1497 ,186  ,1418 ,1822 ,1726 ,947  ,1782 ,1415 ,75   ,1724 ,1769 ,1529 ,1835 ,1262 ,
            834  ,1214 ,685  ,461  ,526  ,1869 ,1373 ,992  ,912  ,1453 ,583  ,652  ,1637 ,798  ,1034 ,1096 ,
            897  ,132  ,1010 ,1932 ,277  ,1536 ,1541 ,952  ,19   ,88   ,2042 ,1232 ,1681 ,2013 ,1241 ,1167 ,
            1526 ,1487 ,761  ,308  ,1567 ,1702 ,177  ,5    ,1709 ,900  ,1699 ,1266 ,1620 ,1027 ,1102 ,1753 ,
            1243 ,471  ,485  ,1765 ,391  ,1281 ,1607 ,1418 ,116  ,1702 ,1725 ,1692 ,1082 ,350  ,14   ,59   ,
            386  ,882  ,2010 ,1438 ,145  ,789  ,1397 ,1921 ,1507 ,457  ,1458 ,1929 ,289  ,1305 ,965  ,500  ,
            1511 ,433  ,284  ,721  ,1741 ,56   ,615  ,916  ,887  ,1253 ,916  ,535  ,1666 ,1175 ,716  ,269  ,
            447  ,32   ,63   ,321  ,1860 ,1986 ,1009 ,1849 ,1062 ,471  ,2018 ,1213 ,1557 ,990  ,696  ,677  ,
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
            // TODO: support both comma (with spaces) and new line
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
