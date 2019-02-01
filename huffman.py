import numpy as np
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
import struct

# Reference: https://github.com/mightydeveloper/Deep-Compression-PyTorch/blob/master/net/huffmancoding.py


test = np.array([1, 2, 2, 3, 5, 6, 8, 0])

Node = namedtuple('Node', ['frequency', 'value', 'left', 'right'])
Node.__lt__ = lambda x, y: x.frequency < y.frequency


def encode_huffman_tree(root):
    """
    Encodes a huffman tree to string of '0's and '1's
    """
    # converter = {'float32':float2bitstr, 'int32':int2bitstr}
    code_list = []

    def encode_node(node):
        if node.value is not None:  # node is leaf node
            code_list.append('1')
            lst = list(int2bitstr(node.value))
            print(lst)
            code_list.extend(lst)
        else:
            code_list.append('0')
            encode_node(node.left)
            encode_node(node.right)

    encode_node(root)
    return ''.join(code_list)


def int2bitstr(integer):
    four_bytes = struct.pack('>I', integer)  # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes)  # string of '0's and '1's


def bitstr2int(bitstr):
    byte_arr = bytearray(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>I', byte_arr)[0]


def huffman_encode(arr):
    # count the frequency of each number in array
    frequency_map = defaultdict(int)
    for value in np.nditer(arr):
        value = int(value)
        frequency_map[value] += 1
    print(frequency_map)

    heap = [Node(frequency, value, None, None) for value, frequency in frequency_map.items()]
    heapify(heap)
    print(heap)

    # Merge nodes
    while len(heap) > 1:
        node1 = heappop(heap)
        node2 = heappop(heap)
        merged = Node(node1.frequency + node2.frequency, None, node1, node2)
        heappush(heap, merged)

    # Generate code value mapping
    value2code = dict()

    def generate_code(node, code):
        if node is None:
            return
        if node.value is not None:
            value2code[node.value] = code
            return
        generate_code(node.left, code + '0')
        generate_code(node.right, code + '1')

    root = heappop(heap)
    generate_code(root, '')

    print(root)
    print(value2code)

    data_encoding = ''.join(value2code[int(value)] for value in np.nditer(arr))
    print(data_encoding)

    codebook_encoding = encode_huffman_tree(root)
    print(codebook_encoding)
    print(type(codebook_encoding))

    return data_encoding, codebook_encoding


data_encoding, codebook_encoding = huffman_encode(test)


def decode_huffman_tree(code_str):
    """
    Decodes a string of '0's and '1's and costructs a huffman tree
    """
    idx = 0

    def decode_node():
        nonlocal idx
        info = code_str[idx]
        idx += 1
        if info == '1':  # Leaf node
            value = bitstr2int(code_str[idx:idx + 32])
            idx += 32
            return Node(0, value, None, None)
        else:
            left = decode_node()
            right = decode_node()
            return Node(0, None, left, right)

    return decode_node()


def huffman_decode(data_encoding, codebook_encoding):
    """
    Decodes binary files from directory
    """

    # Read the codebook
    root = decode_huffman_tree(codebook_encoding)

    # Decode
    data = []
    ptr = root
    for bit in data_encoding:
        ptr = ptr.left if bit == '0' else ptr.right
        if ptr.value is not None:  # Leaf node
            data.append(ptr.value)
            ptr = root

    print(data)
    # return np.array(data, dtype=int)


huffman_decode(data_encoding, codebook_encoding)
