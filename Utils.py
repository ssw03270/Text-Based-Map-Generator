import torch
import json

def map_to_input(map_description, tokenizer):
    """Convert a map description to GPT-2 input format"""
    prefix = "The map description is:\n"
    input_text = prefix + map_description
    input_ids = tokenizer.encode(
        input_text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=512
    )
    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

def output_to_map(output_ids, tokenizer):
    """Convert GPT-2 output format to a 2D map"""
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    map_text = output_text.split("The map is:\n")[1].strip()
    return [list(row) for row in map_text.split('\n')]


def is_empty_map(map_data):
    """Check if the map is empty"""
    for row in map_data:
        for char in row:
            if char == '-':
                return True
    return False

def prompt_generator(map, map_str):
    door_prompt = 'The door is located on the '
    door_location = {'left': False, 'right': False, 'top': False, 'bottom': False}
    door_count = 0
    if map[14][5] =='D':
        door_location['bottom'] = True
        door_count += 1
    if map[1][5] == 'D':
        door_location['top'] = True
        door_count += 1
    if map[7][9] == 'D':
        door_location['right'] = True
        door_count += 1
    if map[7][1] == 'D':
        door_location['left'] = True
        door_count += 1

    if door_count > 2:
        for location in door_location:
            if door_location[location]:
                if door_count > 1:
                    door_prompt += location + ', '
                else:
                    door_prompt += 'and ' + location + '.'
                door_count -= 1

    elif door_count > 1:
        for location in door_location:
            if door_location[location]:
                if door_count > 1:
                    door_prompt += location + ' '
                else:
                    door_prompt += 'and ' + location + '.'
                door_count -= 1

    elif door_count == 1:
        for location in door_location:
            if door_location[location]:
                door_prompt += location + '.'

    else:
        door_prompt = 'There is no door.'

    block_prompt = ''
    block_count = {'No': False, 'Little': False, 'Some': False, 'Many': False}
    if map_str.count('B') == 0:
        block_count['No'] = True
    elif map_str.count('B') < 5:
        block_count['Little'] = True
    elif map_str.count('B') < 10:
        block_count['Some'] = True
    else:
        block_count['Many'] = True

    for count in block_count:
        if block_count[count]:
            block_prompt = count + ' blocks exist.'

    monster_prompt = ''
    monster_count = {'No': False, 'Little': False, 'Some': False, 'Many': False}
    if map_str.count('M') == 0:
        monster_count['No'] = True
    elif map_str.count('M') < 5:
        monster_count['Little'] = True
    elif map_str.count('M') < 10:
        monster_count['Some'] = True
    else:
        monster_count['Many'] = True

    for count in monster_count:
        if monster_count[count]:
            monster_prompt = count + ' monsters exist.'

    element_prompt = ''
    element_count = {'No': False, 'Little': False, 'Some': False, 'Many': False}
    if map_str.count('P') == 0:
        element_count['No'] = True
    elif map_str.count('P') < 10:
        element_count['Little'] = True
    elif map_str.count('P') < 20:
        element_count['Some'] = True
    else:
        element_count['Many'] = True

    for count in element_count:
        if element_count[count]:
            element_prompt = count + ' elements exist.'

    return door_prompt + ' ' + block_prompt + ' ' + monster_prompt + ' ' + element_prompt

def map_changer(map):
    for i in range(len(map)):
        map[i] = map[i].replace('O', 'F')
        map[i] = map[i].replace('I', 'B')

    for i in range(len(map)):
        map[i] = map[i].replace('W', '#')
        map[i] = map[i].replace('F', '~')
        map[i] = map[i].replace('B', '@')
        map[i] = map[i].replace('M', '^')
        map[i] = map[i].replace('P', '*')
        map[i] = map[i].replace('D', '$')

    return map

def join_list_of_list(str_lists):
    return ["".join(s) for s in str_lists]


def characterize(str_lists):
    return [list(s) for s in str_lists]

def map_generator():
    dir_path = './Dataset/MapData/Text/tloz'

    map_data = ''
    for i in range(1, 10):
        for j in range(1, 3):
            file_path = dir_path + str(i) + '_' + str(j) + '.txt'
            # 텍스트 파일 읽기
            with open(file_path, 'r') as f:
                map_data += f.read()

    # 맵 정보 분리
    maps = []
    map_rows = map_data.split('\n')
    map_row_size = 16
    map_col_size = 11
    for i in range(0, len(map_rows), map_row_size):
        for j in range(0, len(map_rows[i]), map_col_size):
            temp_map = []
            for k in range(16):
                temp_map.append(map_rows[i + k][j:j + map_col_size])
            if not is_empty_map(temp_map):
                temp_map = map_changer(temp_map)
                maps.append(temp_map)

    map_data = []
    for i, map in enumerate(maps):
        prompt = prompt_generator(map, '\n'.join(map))
        map_array = characterize(map)
        map_array = "".join(join_list_of_list(map_array))
        map_data.append({
            "prompt": prompt,
            "completion": map_array
        })

    with open('./Dataset/maps.jsonl', 'w') as f:
        json.dump(map_data, f, indent=2)