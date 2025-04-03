import os
import re

benchpress_women_ids = [1,4,6,7,9,13,16,19,34,36,55,66,67,68,69,79,85,99,105, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 
       200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 
       223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238]
deadlift_women_ids = [1,4,6,7,11,12,13,15,18,19,20,21,26,29,31,33,41,44,46,48,50,60,62,63,66,67,68,77,78,79,86,87,88,89,91,92,99,100,102,103,104,105,107,199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 
       222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 
       245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265]
plank_women_ids = [2,3,5,6,7,8,11,12,13,14,15,16,18,21,22,23,27,28,29,32,34,36,37,38,39,40,42,44,45,47,49,50,54,56,58,59,60,63,65,66,67,68,69,70,71,72,73,74,76,79,82,83,85,87,88,91,92,93,94,95,96,97,98,99,100,142]
squat_women_ids = [3,4,10,13,16,17,22,30,37,39,44,46,47,48,51,70,74,76,84,151,173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 
       196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 
       219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 
       242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254]

def count_men(folder_path, women_ids):
    count = 0
    for filename in os.listdir(folder_path):
        match = re.match(r"(\d+)\.jpg$", filename)
        if match:
            number = int(match.group(1))
            if number not in women_ids:
                count += 1
    return count

benchpress_path = "data/bench_press"
deadlift_path = "data/deadlift"
plank_path = "data/plank"
squat_path = "data/squat"

print("Women benchpress: ", len(benchpress_women_ids))
print("Men benchpress: ", count_men(benchpress_path, benchpress_women_ids))
print("Women deadlift: ", len(deadlift_women_ids))
print("Men deadlift: ", count_men(deadlift_path, deadlift_women_ids))
print("Women plank: ", len(plank_women_ids))
print("Men plank: ", count_men(plank_path, plank_women_ids))
print("Women squat: ", len(squat_women_ids))
print("Men squat: ", count_men(squat_path, squat_women_ids))

print("Women total: ", len(squat_women_ids) + len(deadlift_women_ids) + len(plank_women_ids) + len(benchpress_women_ids))
print("Men total: ", count_men(squat_path, squat_women_ids) + count_men(deadlift_path, deadlift_women_ids) + count_men(plank_path, plank_women_ids) + count_men(benchpress_path, benchpress_women_ids))