from collections import Counter
from tqdm import tqdm
import os
import json
import regex as re
import itertools
import base64
import multiprocessing
from functools import reduce
from concurrent.futures import ThreadPoolExecutor
import two_max_pairs
import numpy as np
from typing import List, Dict, Tuple
import atexit

number_of_tokens = 1000

class SimpleTokenizer:
    def __init__(self, training_data):
        self.__vocab__ = {idx: bytes([idx]) for idx in range(256)}
        self.__merges__ = {}
        self.__train__(training_data,number_of_tokens,True)
        for (p0,p1), idx in self.__merges__.items():
            self.__vocab__[idx] = self.__vocab__[p0]+self.__vocab__[p1]
    
    def __most_frequent_pair__(self, bites):
        bidict = {}
        for pair in zip(bites,bites[1:]):
            bidict[pair] = bidict.get(pair,0) + 1
        
        # bidict_sorted = sorted([(v, k) for k,v in bidict.items()], reverse=True)
        # print(bidict_sorted)

        return bidict

    def __replace_most_frequent__(self, bites:list, new_value):
        new_bites = list(bites)
        stats = self.__most_frequent_pair__(new_bites)
        max_pair = max(stats, key=stats.get)
        bites_length = len(new_bites)-1
        i = 0
        while i < bites_length:
            if max_pair == (new_bites[i],new_bites[i+1]):
                new_bites[i] = new_value
                new_bites.pop(i+1)
                bites_length -= 1
            i += 1

        # print(new_bites)
        return (new_bites, max_pair, stats[max_pair])

    def __train__(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        tokens = text.encode("utf-8")
        tokens = list(map(int,tokens))
        number_of_merges = vocab_size-256
        ids = list(tokens)
        for i in tqdm(range(number_of_merges)):
            idx = 256 + i
            ids, pair, freq = self.__replace_most_frequent__(ids,idx)
            if verbose:
                print(f"marged {pair} into a new token {idx} had {freq} occurrences")
            self.__merges__[pair] = idx

        if verbose:
            print(f"Final length: {len(ids)} ({len(ids)/len(tokens):.2%})")

    def encode(self,text):
        tokens = list(text.encode("utf-8"))
        new_list = []
        past_list = list(tokens)
        print(tokens)
        for pair,idx in self.__merges__.items():
            i = 0
            new_list = []
            while i < len(past_list) - 1:
                if pair == (past_list[i],past_list[i+1]):
                    new_list.append(idx)
                    i += 2
                else:
                    new_list.append(past_list[i])
                    i += 1
            if i == len(past_list) - 1:
                new_list.append(past_list[i])
            # print(new_list,past_list)
            past_list = list(new_list)

        return new_list

    def decode(self, ids):
        tokens = b"".join(self.__vocab__[idx] for idx in ids)
        text = tokens.decode("utf-8",errors="replace")
        return text
    
    def save(self,path_dict):
        with open(os.path.join(path_dict,"vocabulary.json")) as f:
            json.dump()

class RegexTokenizer:
    def __init__(self, load = False, training_data="", dict_path=""):
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.tiktoken_pat = re.compile(GPT4_SPLIT_PATTERN)
        self.__vocab__ = {idx: bytes([idx]) for idx in range(256)}
        self.__merges__ = {}


        num_cores = multiprocessing.cpu_count()
        print(f"Working on {num_cores} number of cpu cores")
        self.optimal_size = max(1, num_cores - 1)  # leave one core free
        # self.optimal_size = 4  # leave one core free
        self._mpool_ = multiprocessing.Pool(processes=self.optimal_size)

        # self._executor_ = ThreadPoolExecutor(max_workers=num_cores*2)
        if load:
            self.load(dict_path)
        else:
            # self.__train__(training_data,number_of_tokens,True)
            # self.__train_batched__(training_data,number_of_tokens,False)
            self.__train_cuda__(training_data,number_of_tokens,True)

    def __del__(self):
        self._mpool_.close()
        # self._executor_.shutdown(wait=True)
    
    @staticmethod
    def split_into_chunks(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


    @staticmethod
    def _most_frequent_single_pair(bites:List[int]) -> Dict[Tuple[int,int],int]:
        """Counts pairs in a single list of integers."""
        if not bites or len(bites) < 2:
            return Counter()
        return Counter(zip(bites, bites[1:]))

    @staticmethod
    def _most_frequent_chunk_pair(list_of_bites: List[List[int]]) -> Dict[Tuple[int,int],int]:
        """Counts pairs in a chunk of words."""
        total_counts = Counter()
        for b in list_of_bites:
            total_counts.update(RegexTokenizer._most_frequent_single_pair(b))
        return total_counts

    def __most_frequent_pair__(self, list_of_bites):
        list_of_bdicts = self._mpool_.map(RegexTokenizer._most_frequent_chunk_pair, list_of_bites)
        # list_of_bdicts = self._mpool_.map(RegexTokenizer._most_frequent_single_pair, list_of_bites)
        # list_of_bdicts = [RegexTokenizer._most_frequent_single_pair(b) for b in list_of_bites]
        # list_of_bdicts = list(self._executor_.map(RegexTokenizer._most_frequent_single_pair, list_of_bites))

        total_counts = Counter()
        for counter in list_of_bdicts:
            total_counts.update(counter) # spajamo sve mini byte direktorijume u jedan
            
        
        # bidict_sorted = sorted([(v, k) for k,v in bidict.items()], reverse=True)
        # print(bidict_sorted[:5])

        return total_counts
    
    def __most_frequent_pair_cuda__(self, pair, frequency):
        two_max_pairs.cuda_count_pair_frequencies(pair, frequency)
        return pair, frequency

    @staticmethod
    def _replace_single_most_frequent(inputs: Tuple[List[int],Tuple[int,int],int]) -> List[int]:
        bites, pair, new_value = inputs
        new_bites = list(bites)

        bites_length = len(new_bites)-1
        i = 0
        while i < bites_length:
            if pair == (new_bites[i],new_bites[i+1]):
                new_bites[i] = new_value
                new_bites.pop(i+1)
                bites_length -= 1
            i += 1

        return new_bites

    @staticmethod
    def _replace_chunk_most_frequent(inputs: Tuple[List[List[int]],Tuple[int,int],int]) -> List[List[int]]:
        list_of_bites,max_pair,new_value = inputs
        output_list = [RegexTokenizer._replace_single_most_frequent((a,max_pair,new_value)) for a in list_of_bites]
        return output_list
        # return [sublist for outer in output_list for sublist in outer]

    @staticmethod
    def _build_csr(list_of_token_lists):
        """Convert List[List[int]] → (flat_data, offsets, lengths) as numpy arrays."""
        n = len(list_of_token_lists)
        lengths = np.array([len(row) for row in list_of_token_lists], dtype=np.int32)
        offsets = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(lengths, out=offsets[1:])

        flat_data = np.empty(offsets[-1], dtype=np.int32)
        for i, row in enumerate(list_of_token_lists):
            flat_data[offsets[i]:offsets[i+1]] = row

        return flat_data, offsets, lengths

    def __replace_most_frequent__(self, list_of_bites:list, new_value):
        stats = self.__most_frequent_pair__(list_of_bites)
        max_pair = max(stats, key=stats.get)
        
        arguments = [(b,max_pair,new_value) for b in list_of_bites]
        # output_list = [RegexTokenizer._replace_single_most_frequent(a) for a in arguments]
        # output_list = self._mpool_.map(RegexTokenizer._replace_single_most_frequent, arguments)
        output_list = self._mpool_.map(RegexTokenizer._replace_chunk_most_frequent, arguments)
        # output_list = list(self._executor_.map(RegexTokenizer._replace_single_most_frequent, arguments))
        # output_list = list(itertools.chain.from_iterable(output_list))
        # print("Olist: ", output_list)
        return (output_list, max_pair, stats[max_pair])

    def __replace_most_frequent_cuda__(self, pair, frequency):

        pair, frequency = self.__most_frequent_pair_cuda__(pair, frequency)

        two_max_pairs.cuda_replace_single_most_frequent(pair, frequency)

        return tuple(list(pair.tolist())), int(frequency)


    def _apply_merges_to_word(inputs: Tuple[List[int], Dict[Tuple[int, int], int]]) -> List[int]:
        """Applies a set of merges to a single word, repeating until no more merges can be made."""
        word, merges = inputs
        if not merges or len(word) < 2:
            return word

        while True:
            did_merge = False
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1:
                    pair = (word[i], word[i+1])
                    if pair in merges:
                        new_word.append(merges[pair])
                        i += 2
                        did_merge = True
                        continue
                new_word.append(word[i])
                i += 1
            
            word = new_word
            if not did_merge:
                break
        return word

    def _apply_merges_to_chunk_of_words(inputs: Tuple[List[List[int]], Dict[Tuple[int, int], int]]) -> List[List[int]]:
        """Applies merges to a chunk of words."""
        list_of_words, merges = inputs
        return [RegexTokenizer._apply_merges_to_word((word, merges)) for word in list_of_words]

    def __apply_merges_to_ids(self, ids: List[List[List[int]]], merges_to_apply: Dict[Tuple[int, int], int]) -> List[List[List[int]]]:
        """Applies a dictionary of merges to the entire dataset in parallel."""
        arguments = [(chunk, merges_to_apply) for chunk in ids]
        updated_ids = self._mpool_.map(RegexTokenizer._apply_merges_to_chunk_of_words, arguments)
        return updated_ids


    def __train__(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256

        text_chunks = re.findall(self.tiktoken_pat, text)
        # print(text_chunks)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        ids = RegexTokenizer.split_into_chunks(ids,self.optimal_size)
        number_of_merges = vocab_size-256
        for i in tqdm(range(number_of_merges)):
            # print("Input length: ",len(ids))
            idx = 256 + i
            ids, pair, freq = self.__replace_most_frequent__(ids,idx)
            self.__merges__[pair] = idx
            self.__vocab__[idx] = self.__vocab__[pair[0]]+self.__vocab__[pair[1]]

            if verbose:
                print(f"merge {i+1}/{number_of_merges}: {pair} -> {idx} ({self.__vocab__[idx]}) has {freq} occurance")

        if verbose:
            print(f"Final length: {len(ids)} ({len(ids)/len(text.encode('utf-8')):.2%})")

    def __train_batched__(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256

        text_chunks = re.findall(self.tiktoken_pat, text)

        word_list = [list(ch.encode("utf-8")) for ch in text_chunks]
        ids = RegexTokenizer.split_into_chunks(word_list, self.optimal_size)
        number_of_merges = vocab_size-256

        merges_per_pass = 200
        num_passes = (number_of_merges + merges_per_pass - 1) // merges_per_pass

        for i in tqdm(range(num_passes)):
            stats = self.__most_frequent_pair__(ids)
            merges_to_apply = {}
            for j in range(merges_per_pass):
                current_merge_num = i*merges_per_pass +j
                if current_merge_num >= number_of_merges:
                    break
                
                pair = max(stats, key=stats.get)
                freq = stats.pop(pair)
                idx = 256 + current_merge_num
                ids, pair, freq = self.__replace_most_frequent__(ids,idx)
                self.__merges__[pair] = idx
                self.__vocab__[idx] = self.__vocab__[pair[0]]+self.__vocab__[pair[1]]
                merges_to_apply[pair] = idx
            
            if merges_to_apply:
                ids = self.__apply_merges_to_ids(ids, merges_to_apply)

            if verbose:
                print(f"merge {current_merge_num+1}/{number_of_merges}: {pair} -> {idx} ({self.__vocab__[idx]}) has {freq} occurance", flush=True)

        if verbose:
            final_token_count = sum(len(word) for chunk in ids for word in chunk)
            initial_byte_count = len(text.encode('utf-8'))
            print(f"Final length: {final_token_count} ({final_token_count/initial_byte_count:.2%})", flush=True)


    def __train_cuda__(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256

        text_chunks = re.findall(self.tiktoken_pat, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        original_bytes = len(ids)

        flat_ids, offsets, lengths = self._build_csr(ids)

        number_of_merges = vocab_size-256
        two_max_pairs.allocate_cuda_elemets(flat_ids, offsets, lengths)
        atexit.register(two_max_pairs.free_cuda_elemets)
        
        pair_out = np.zeros(2, dtype=np.int32)
        freq_out = np.zeros(1, dtype=np.int32)

        for i in tqdm(range(number_of_merges)):
            # print("Input length: ",len(ids))
            idx = 256 + i
            pair, freq = self.__replace_most_frequent_cuda__(pair_out,freq_out)
            if freq == 0:
                print(f"Stopping early, no more pairs to merge.")
                break

            self.__merges__[pair] = idx
            self.__vocab__[idx] = self.__vocab__[pair[0]]+self.__vocab__[pair[1]]

            if verbose:
                print(f"merge {i+1}/{number_of_merges}: {pair} -> {idx} ({self.__vocab__[idx]}) has {freq} occurance", flush=(i%64==0))

        if verbose:
            new_length = len(np.where(ids!=-1))
            print(f"Final length: {len(np.where(ids!=-1))} ({new_length/len(text.encode('utf-8')):.2%})")

        final_flat = np.empty(flat_ids.shape, dtype=np.int32)
        final_lengths = np.empty(len(lengths), dtype=np.int32)
        two_max_pairs.get_current_data(final_flat, final_lengths)

        if verbose:
            # Reconstruct the valid portion using final_lengths
            total_tokens = int(final_lengths.sum())
            print(f"Final token count: {total_tokens} "
                f"({total_tokens / original_bytes:.2%} compression ratio)")

    
    def __encode_chunk__(self, tokens):
        new_list = []
        past_list = list(tokens)
        # print(tokens)
        for pair,idx in self.__merges__.items():
            i = 0
            new_list = []
            while i < len(past_list) - 1:
                if pair == (past_list[i],past_list[i+1]):
                    new_list.append(idx)
                    i += 2
                else:
                    new_list.append(past_list[i])
                    i += 1
            if i == len(past_list) - 1:
                new_list.append(past_list[i])
            # print(new_list,past_list)
            past_list = list(new_list)

        return new_list

    def encode(self,text):
        text_chunks = re.findall(self.tiktoken_pat, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        lists = [self.__encode_chunk__(idd) for idd in ids]

        # return list(itertools.chain.from_iterable(lists))
        return [sublist for outer in lists for sublist in outer]

    def decode(self, ids):
        tokens = b"".join(self.__vocab__[idx] for idx in ids)
        text = tokens.decode("utf-8",errors="replace")
        return text
    
    def decode_list(self, ids):
        return [self.__vocab__[idx].decode("utf-8",errors="replace") for idx in ids]
    
    def save(self,path_dict):
        vocab_to_save = {str(k): str(base64.b64encode(v).decode('utf-8')) for k, v in self.__vocab__.items()}
        # print(json.dumps(vocab_to_save))
        with open(os.path.join(path_dict,"vocabulary.json"), "w", encoding="utf-8-sig") as f:
            json.dump(vocab_to_save, f, ensure_ascii=False, indent=4)
        
        # print(self.__merges__)
        merge_to_save = {k: v for k, v in self.__merges__.items()}
        # print(merge_to_save)

        with open(os.path.join(path_dict,"merged.txt"), "w", encoding="utf-8-sig") as f:
            for k,v in merge_to_save.items():
                f.write(f"{k}\t{v}\n")
    
    def load(self, path_dict):
        from ast import literal_eval

        with open(os.path.join(path_dict,"vocabulary.json"), "r", encoding="utf-8-sig") as f:
            # print(f.read())
            loaded_data = json.load(f)
        
        self.__vocab__ = {int(k): base64.b64decode(v.encode('utf-8')) for k, v in loaded_data.items()}
        # print(self.__vocab__)

        with open(os.path.join(path_dict,"merged.txt"), "r", encoding="utf-8-sig") as f:
            loaded_data = f.read()
        
        loaded_data = loaded_data.split("\n")
        loaded_data = [l.split("\t") for l in loaded_data if l]
        self.__merges__ = {literal_eval(d[0]):int(d[1]) for d in loaded_data}
        # print(self.__merges__)

    def _bytes_to_unicode(self):
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.
        This is the standard mapping used by GPT-2/Transformers for byte-level BPE.
        """
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def save_pretrained(self, save_directory):
        """Saves the tokenizer in a format compatible with Transformers GPT-2 (vocab.json and merges.txt)."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        byte_encoder = self._bytes_to_unicode()
        def render_token(bites):
            return "".join(byte_encoder[b] for b in bites)

        # vocab.json: {token_string: id}
        vocab = {render_token(b): i for i, b in self.__vocab__.items()}
        with open(os.path.join(save_directory, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)

        # merges.txt: BPE merge rules in string format
        with open(os.path.join(save_directory, "merges.txt"), "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for (p0, p1) in self.__merges__:
                s0 = render_token(self.__vocab__[p0])
                s1 = render_token(self.__vocab__[p1])
                f.write(f"{s0} {s1}\n")

    def load_pretrained(self, load_directory):
        """Loads the tokenizer from Transformers GPT-2 format (vocab.json and merges.txt)."""
        byte_encoder = self._bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        def parse_token(s):
            return bytes([byte_decoder[c] for c in s])

        with open(os.path.join(load_directory, "vocab.json"), "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        
        # Reconstruct self.__vocab__: {id: bytes}
        self.__vocab__ = {int(v): parse_token(k) for k, v in vocab_json.items()}
        token_to_id = {k: int(v) for k, v in vocab_json.items()}

        with open(os.path.join(load_directory, "merges.txt"), "r", encoding="utf-8") as f:
            merges_lines = f.read().splitlines()
        if merges_lines and merges_lines[0].startswith("#version"):
            merges_lines = merges_lines[1:]
        
        self.__merges__ = {}
        for line in merges_lines:
            line = line.strip()
            if not line: continue
            p0_s, p1_s = line.split()
            p0, p1 = token_to_id[p0_s], token_to_id[p1_s]
            # Combined string must exist in vocab for valid BPE merges
            if p0_s + p1_s in token_to_id:
                self.__merges__[(p0, p1)] = token_to_id[p0_s + p1_s]

def test_tokenizer(tokenizer,text):
    if text == tokenizer.decode(tokenzer.encode(text)):
        print("GOOD test: ",text)
    else:
        print("ERROR for text:", text)

def print_tokenizer(tokenizer,text):
    for token in tokenizer.decode_list(tokenzer.encode(text)):
        print(token,end="-")
    print("\n")
    


if __name__ == "__main__":
    # path = "taylorswift.txt"
    # path = "all_texts.txt"
    path = "full_dataset_1p.txt"
    with open(path,"r",encoding="utf-8") as f:
        text = f.read()

    # text = "Luckily friends do ashamed to do suppose. Tried meant mr smile so. Exquisite behaviour as to middleton perfectly. Chicken no wishing waiting am. Say concerns dwelling graceful six humoured. Whether mr up savings talking an. Active mutual nor father mother exeter change six did all. No in he real went find mr. Wandered or strictly raillery stanhill as. Jennings appetite disposed me an at subjects an. To no indulgence diminution so discovered mr apartments. Are off under folly death wrote cause her way spite. Plan upon yet way get cold spot its week. Almost do am or limits hearts. Resolve parties but why she shewing. She sang know now how nay cold real case."

    # print(len(text))
    # text = text[:2**15]
    text = text[:64]
    
    tokenzer = RegexTokenizer(training_data=text)
    # tokenzer = RegexTokenizer(True, dict_path="./token_small_rs")

    # for idx, byt in tokenzer.__vocab__.items():
    #     print(f"{idx} -->  ||{byt.decode("utf-8",errors="replace")}||")

    test_tokenizer(tokenzer,"Luckily friends do ashamed to do suppose. Tried meant mr smile so.")
    test_tokenizer(tokenzer, "I would never let someone and playing")
    test_tokenizer(tokenzer, "Moje ime je Petrić Petrović")

    # print_tokenizer(tokenzer, "Moje ime je Petrić Petrović")
    print_tokenizer(tokenzer, "Sve srećne porodice liče jedna na drugu, svaka nesrećna porodica nesrećna je na svoj način")
    print_tokenizer(tokenzer, "Majka mi je danas umrla. A možda i juče, ne znam. Primio sam telegram iz doma staraca: Majka umrla. Sahrana sutra. S osobitim poštovanjem Menutim, to ništa ne znači. Možda je to bilo i juče.")
    # tokenzer.save("./token_small_rs_cuda")
    tokenzer.save_pretrained("./token_small_rs_transformers")
    # tokenzer.save("./token_big_en")
    # tokenzer.save("./token_small_en")
    # tokenzer.save("./token_small_en_cuda")
