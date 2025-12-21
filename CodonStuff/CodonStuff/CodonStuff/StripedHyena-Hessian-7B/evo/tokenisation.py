# The purpose of this Python script is to define the functions required
# for tokenising amino acid as well as nucleotide sequences

from math import ceil
import torch
import json
from torch.nn.utils.rnn import pad_sequence
import Levenshtein


# import tensorflow as tf
import numpy as np
# import keras_nlp
class Tokenizer:
    def __init__(self):

        # Load unified vocab
        vocab_path = "/home/rm15weti/CodonStuff/CodonStuff/CodonStuff/StripedHyena-Hessian-7B/evo/tokenizer/full_biotokenizer.json"
        with open(vocab_path, "r") as f:
            self.token_to_index = json.load(f)

        self.index_to_token = {v: k for k, v in self.token_to_index.items()}
        self.vocab_size = max(self.token_to_index.values()) + 1

        # Special token IDs (if needed explicitly)
        self.pad_token = "<PAD>"
        self.sep_token = "<SEP>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"

        self.pad_token_id = self.token_to_index.get(self.pad_token)
        self.sep_token_id = self.token_to_index.get(self.sep_token)
        self.start_token_id = self.token_to_index.get(self.start_token)
        self.end_token_id = self.token_to_index.get(self.end_token)
        self.unk_token_id = self.token_to_index.get(self.unk_token)

        # Optional, if you're distinguishing AA and codon tokens:
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY*'
        self.codons = [nt1 + nt2 + nt3 for nt1 in "ATCG" for nt2 in "ATCG" for nt3 in "ATCG"]

        # Derived vocab segments (use only if needed)
        self.aa_tokens = [aa for aa in self.amino_acids if aa in self.token_to_index]
        self.codon_tokens = [codon for codon in self.codons if codon in self.token_to_index]
        self.special_tokens = [tok for tok in [self.pad_token, self.start_token, self.end_token, self.sep_token, self.unk_token] if tok in self.token_to_index]      


    def parse_seq(self, seq):
        """
        This function is responsible for converting an input sequence to a
        string, if applicable, and to return the input string otherwise.

        Parameters
        ----------
        seq: str or binary
            The input amino acid sequence to be tokenised.

        Returns
        -------
        seq: str
            The input amino acid sequence to be tokenised.
        """
        if isinstance(seq, str):
            return seq
        elif isinstance(seq, bytes):
            return seq.decode('utf8')
        else:
            raise TypeError('Unexpected sequence type: %s' % type(seq))


    def tokenise_aa_seq(self, seq, add_special_tokens=True):
        """
        This function performs tokenisation for amino acid sequences. This
        is achieved by splitting an amino acid sequence into individual
        tokens, i. e. individual amino acids, and replacing all individual 
        tokens with their unique integer index defined in the dictionary
        `token_to_index`.

        Apart from that, one <START> and <END> token is added to each
        sequence, respectively. The token dictionary also comprises a
        special token, <OTHER>, reserved for unknown amino acids.

        This function employs the private function `parse_seq`.

        Parameters
        ----------
        seq:  str or binary
            The amino acid sequence to be tokenised.

        Returns
        -------
        tokenised_seq: list, dtype=int
            The tokenised analogue of the input amino acid sequence.
        """
        tokens = [
            self.token_to_index.get(aa)
            for aa in self.parse_seq(seq)
        ]
        if add_special_tokens:
            tokens = (
                [self.start_token_id]
                + tokens
                #+ [self.end_token_id]
            )
        return tokens


    def tokenise_aa_seqs(self, seqs):
        """
        This function performs tokenisation for a whole data set of amino
        acid sequences.

        For this purpose, it employs the private function
        `tokenise_aa_seq`.

        Parameters
        ----------
        seqs: iterable, dtype=str or dtype=binary
            An iterable containing the amino acid sequences to be tokenised.

        Returns
        -------
        tokenised_seqs_array: ndarray, dtype=int32, shape=(m, n)
            A NumPy array harbouring the tokenised analogues of the input
            amino acid sequences. The first dimension, m, corresponds to the
            amount of input sequences, whereas the second dimension, n,
            corresponds to the largest sequence length in the batch.
        """
        # First, perform the mere tokenisation
        tokenised_seqs_list = []

        for seq in seqs:
            for token in seq:
                index = self.token_to_index.get(token)
                if index is None:
                    raise ValueError(f"Unknown amino acid token: {token} in sequence {seq}")
                    break
            tokenised_seqs_list.append(self.tokenise_aa_seq(seq))

        max_length = max(map(len, tokenised_seqs_list))

        # Perform padding for each of the tokenised sequences
        tokenised_seqs_array = np.array(
            [tokenised_seq
            +
            (max_length - len(tokenised_seq))*[self.pad_token_id]
            for tokenised_seq in tokenised_seqs_list],
            dtype=np.int32
            )

        #return tokenised_seqs_array
        return tokenised_seqs_list


    @property
    def vocab_size_NT(self):
        # Example for codons
        return len(self.ALL_CODONS) + len(self.ADDITIONAL_TOKENS)
    
    def _split_into_triplets(self, nt_seq):
        """
        This function splits an input nucleotide sequence into its
        individual codons / triplets.

        The function assumes that the sequence length is a multiple of three
        and that the first and last codon is the AUG start codon and one of
        the three stop codons, respectively.

        Parameters
        ----------
        nt_seq: str
            The nucleotide sequence to be split into triplets / codons.

        Returns
        -------
        codons: list, dtype=str
            The list containing the codons the input nucleotide sequence is
            made up of.
        """
        codons = []
        n_triplets = len(nt_seq) // 3

        for i in range(n_triplets):
            start_index = 3 * i
            end_index = 3 * (i + 1)
            triplet = nt_seq[start_index:end_index]
            codons.append(triplet)

        return codons


    def _tokenise_nt_seq(self, seq, add_special_tokens=True):
        """
        This function is analogous to the function `tokenise_aa_seq` and
        performs tokenisation for nucleotide sequences. However, instead of
        taking the individual nucleotide letters as tokens and mapping them
        to integer indices, three successive nucleotides, i. e. triplets /
        codons are taken as tokens. Hence, there are 64 unique tokens,
        without the additional tokens (e. g. <START>, <END>, etc.).

        For this purpose, it employs the private function
        `_split_into_triplets`.

        Parameters
        ----------
        seq: str or binary
            The input nucleotide sequence to be tokenised.

        Returns
        -------
        tokenised_seq: list, dtype=int
            The tokenised analogue of the input nucleotide sequence.
        """
        tokens = [
            self.token_to_index.get(codon)
            for codon in self._split_into_triplets(self.parse_seq(seq))
        ]
        if add_special_tokens:
            tokens = (
                #[self.start_token_id]
                tokens
                + [self.end_token_id]
            )
        return tokens


    def _tokenise_nt_seqs(self, seqs):
        """
        This function performs tokenisation for a whole data set of
        nucleotide sequences.

        For this purpose, it employs the private function 
        `_tokenise_nt_seq`.

        Parameters
        ----------
        seqs: iterable, dtype=str or dtype=binary
            An iterable containing the nucleotide sequences to be tokenised.
        seq_len: int
            Integer indicating the maximum sequence length, but in terms of
            codons / triplets instead of in terms of single nucleotides.
            Input sequences shorter than this value are padded with a
            special token until they reach the specified length. When
            passing this value to the function call, it should be taken into
            consideration that the <START> and <END> tokens are already
            added to each individual sequence. Hence, in order to avoid a
            ValueError, the sequence length needs to be incremented by 2.

            Apart from that, the tokenised nucleotide sequence is shortened
            by one unit at the beginning and the end later on, respectively.
            Hence, its length must be greater by 1 than the length of the
            corresponding amino acid sequence. This, however, is handled by
            the function itself, so that the user can insert the same
            sequence length for both the amino acid and the nucleotide
            sequence.

        Returns
        -------
        tokenised_seqs_array: ndarray, dtype=int32, shape=(m, n)
            A NumPy array harbouring  the tokenised analogues of the input
            nucleotide sequences. The first dimension, m, corresponds to the
            amount of input sequences, whereas the second dimension, n,
            corresponds to the largest sequence length of the batch.
        """
        # First, perform the mere tokenisation
        tokenised_seqs_list = []

        for seq in seqs:
            tokenised_seqs_list.append(self._tokenise_nt_seq(seq))
        max_length = max(map(len, tokenised_seqs_list))
        max_length += 1

        tokenised_seqs_array = np.array(
            [tokenised_seq
            +
            (
                max_length - len(tokenised_seq)
            )*[self.pad_token_id]
            for tokenised_seq in tokenised_seqs_list],
            dtype=np.int32
        )

        #return tokenised_seqs_array
        return tokenised_seqs_list


    def tokenise_nt_seqs_for_eval(self, seqs, max_seq_len):
        """
        This function performs tokenisation for a whole data set of
        nucleotide sequences. For this purpose, it employs the private
        function '_tokenise_nt_seq'.

        The difference between this function and '_tokenise_nt_seqs' is that
        this function has an additional parameter, 'max_seq_len', indicating
        the maximum sequence length in terms of codons/triplets. This is
        opposed to the behaviour of the function '_tokenise_nt_seqs', which
        adopts the length of the largest input sequence as the maximum
        sequence length.
        
        Due to this implementation, 'max_seq_len' must be at least equal to
        the length of the largest input sequence. Otherwise, a ValueError is
        raised.

        Parameters
        ----------
        seqs: iterable, dtype=str or dtype=binary
            An iterable containing the nucleotide sequences to be tokenised.
        max_seq_len: int
            An integer indicating the maximum sequence length, but in terms
            of codons/triplets instead of in terms of single nucleotides.
            Input sequences shorter than this value are padded with a
            special token until they reach the specified length. When
            passing this value to the function call, it should be taken into
            consideration that the <START> and <END> tokens are already
            added to each individual sequence. Hence, in order to avoid a
            ValueError, the sequence length needs to be incremented by 2.

            Apart from that, the argument passed to this parameter needs to
            be at least equal to the length of the largest input sequence.
            Otherwise, a ValueError is raised.

        Returns
        -------
        tokenised_seqs_array: ndarray, dtype=int32, shape=(m, n)
            A NumPy array harbouring the tokenised analogues of the input
            nucleotide sequences. The first dimension, m, corresponds to the
            amount of input sequences, whereas the second dimension, n,
            corresponds to the maximum sequence length.
        """
        max_input_length = max(map(len, seqs)) / 3
        if max_seq_len < max_input_length:
            raise ValueError(
                "The argument passed to 'max_seq_len' must be at least "
                "equal to the length of the largest input sequence!"
            )
        
        tokenised_seqs_list = []

        for seq in seqs:
            tokenised_seqs_list.append(self._tokenise_nt_seq(seq))
        
        # Perform padding for each of the tokenised sequences
        tokenised_seqs_array = np.array(
            [tokenised_seq
            +
            (
                max_seq_len - len(tokenised_seq)
            ) * [self.pad_token_id]
            for tokenised_seq in tokenised_seqs_list],
            dtype=np.int32
        )

        return tokenised_seqs_array


    def tokenise_aa_nt_pair(self, aa_seqs, nt_seqs):
        """
        Tokenises and concatenates AA and NT sequences with special tokens.

        Parameters:
        - aa_seqs: list of amino acid sequences (str)
        - nt_seqs: list of nucleotide sequences (str)

        Returns:
        - tokenised_pairs: Tensor of shape (batch_size, sequence_len)
        """

        assert len(aa_seqs) == len(nt_seqs), "AA and NT sequence batch sizes must match"

        # Tokenise each individually
        aa_tokens = self.tokenise_aa_seqs(aa_seqs)     # shape: (B, L_aa)
        #print(f"AA tokens: {aa_tokens}")
        nt_tokens = self._tokenise_nt_seqs(nt_seqs)    # shape: (B, L_nt)
        #print(f"NT tokens: {nt_tokens}")
        START = self.start_token_id
        END = self.end_token_id
        SEP = self.sep_token_id 
        batch_size = len(aa_seqs)

        combined_tokens = []

        for i in range(batch_size):
            aa_seq = aa_tokens[i]
            nt_seq = nt_tokens[i]

            combined = np.concatenate([
                np.array([START], dtype=np.int32),
                aa_seq, 
                np.array([SEP], dtype=np.int32), 
                nt_seq,
                np.array([END], dtype=np.int32)
            ])
            combined_tokens.append(combined)

        # Efficient conversion: if all same length, stack; if single sample, unsqueeze; else fallback
        try:
            lengths = {len(x) for x in combined_tokens}
            if len(lengths) == 1:
                import numpy as _np
                return torch.from_numpy(_np.stack(combined_tokens)).long()
            elif len(combined_tokens) == 1:
                import numpy as _np
                return torch.from_numpy(_np.expand_dims(combined_tokens[0], 0)).long()
            else:
                return torch.tensor(combined_tokens, dtype=torch.long)  # ragged fallback (may warn)
        except Exception:
            return torch.tensor(combined_tokens, dtype=torch.long)

    def _detokenise_nt_seqs_torch(self, nt_token_tensor):
        """
        This function performs detokenisation for a whole data set of
        nucleotide sequences using PyTorch tensors.

        Parameters
        ----------
        nt_token_tensor: torch.Tensor, dtype=torch.int
            A tensor containing the sequences of token IDs generated in the
            course of translation.

        Returns
        -------
        nt_seq_array: NumPy array, dtype=str
            A one-dimensional NumPy array containing the sequences generated
            during translation as strings. In other words, it contains the
            generated sequences as human-readable nucleotide sequences.
        """
        # Convert the PyTorch tensor into a NumPy array for easier handling
        nt_token_array = nt_token_tensor.cpu().numpy()

        nt_seq_list = []

        for token_seq in nt_token_array:
            # Map each token ID to its corresponding codon using the dictionary
            codon_list = [
                self.index_to_codon_token.get(token)
                for token in token_seq
            ]
            # Join the codons into a single nucleotide sequence string
            codon_str = "".join(codon_list)
            nt_seq_list.append(codon_str)

        # Convert the list into a NumPy array
        nt_seq_array = np.array(nt_seq_list, dtype=str)

        return nt_seq_array

    def _detokenise_nt_seqs_torch(self, nt_token_tensor):
        """
        Converts token IDs to nucleotide sequences, stripping <END> and <PAD> tokens.

        Args:
            nt_token_tensor (List[List[int]] | Tensor | np.ndarray)

        Returns:
            List[str]: List of nucleotide sequences (e.g. ['ATGGCC...', ...])
        """
        if isinstance(nt_token_tensor, torch.Tensor):
            nt_token_tensor = nt_token_tensor.detach().cpu().tolist()
        elif isinstance(nt_token_tensor, np.ndarray):
            nt_token_tensor = nt_token_tensor.tolist()

        sequences = []
        for row in nt_token_tensor:
            codons = []
            for token_id in row:
                if token_id == self.end_token_id or token_id == self.pad_token_id:
                    break
                codon = self.index_to_token.get(token_id, '')  # handle unknown token_ids
                codons.append(codon)
            sequences.append(''.join(codons))

        return sequences

    def format_dataset_torch(self, aa_nt_pairs, max_seq_length=None, device='cpu'):
        """
        Build *unpadded* input_id sequences and label sequences for AA<SEP>NT training.
        Padding and attention_mask will be created later in collate_fn.

        Returns:
            input_id_list: List[List[int]]
            label_list:    List[List[int]]  (with -100 on AA+SEP positions)
        """
        # Convert DF -> arrays of strings
        aa_nt_pairs = aa_nt_pairs.to_numpy()
        aa_seqs, nt_seqs = zip(*aa_nt_pairs)

        input_id_list = []
        label_list = []

        for aa, nt in zip(aa_seqs, nt_seqs):
            # Tokenise and concatenate: <START> AA ... <SEP> NT ... <END>
            toks = self.tokenise_aa_nt_pair([aa], [nt])  # Tensor [1, L] or ragged fallback
            toks = toks.flatten().tolist()

            # Sanity: require <SEP>
            if self.sep_token_id not in toks:
                continue

            # (Optional) truncate if a hard cap is desired (keeps sequence valid)
            if max_seq_length is not None and len(toks) > max_seq_length:
                # Keep tail so we don't drop <END>; simple strategy — you can tailor this
                toks = toks[:max_seq_length]

            sep_idx = toks.index(self.sep_token_id)

            # Labels: ignore AA + <SEP>, learn on codon positions
            labels = [(-100 if i <= sep_idx else tok) for i, tok in enumerate(toks)]

            input_id_list.append(toks)
            label_list.append(labels)

        if len(input_id_list) == 0:
            raise ValueError("No valid AA<SEP>NT samples were produced (missing <SEP> or empty input).")

        # NOTE: do NOT make tensors or move to device here. Let collate_fn handle padding and device.
        return input_id_list, label_list


    def translate_aa_into_nt_torch(
        self,
        transformer,
        aa_seqs,
        max_seq_length,
        return_string=True,
        batch_size=32,
        device="cpu",
        temperature=0.5,
    ):
        """
        Translate AA sequences -> NT codon sequences with a causal LM trained on:
        <START> AAs <SEP> NTs <END>

        Sampling is restricted to {codon tokens} ∪ {<END>}.
        """
        # Tokenize & pad AA side (no <START>/<SEP> injected here)
        aa_ids = self.tokenise_aa_seqs(aa_seqs)
        aa_ids = [torch.tensor(s, dtype=torch.long) for s in aa_ids]
        aa_pad = pad_sequence(aa_ids, batch_first=True, padding_value=self.pad_token_id).to(device)

        n = aa_pad.size(0)
        n_batches = (n + batch_size - 1) // batch_size
        outputs = []

        start_id = self.start_token_id
        sep_id   = self.sep_token_id
        end_id   = self.end_token_id
        pad_id   = self.pad_token_id

        # If codon IDs are contiguous and start after AA/specials:
        aa_vocab_max   = self.token_to_index['*']      # last AA/special in your scheme
        first_codon_id = aa_vocab_max + 1              # assumes contiguous codon block

        transformer.eval()
        with torch.no_grad():
            for bi in range(n_batches):
                batch_aa = aa_pad[bi*batch_size : (bi+1)*batch_size]
                B = batch_aa.size(0)

                # Prefix: <START> AAs <SEP>
                start = torch.full((B, 1), start_id, dtype=torch.long, device=device)
                sep   = torch.full((B, 1), sep_id,   dtype=torch.long, device=device)
                prefix = torch.cat([start, batch_aa, sep], dim=1)

                generated = torch.empty(B, 0, dtype=torch.long, device=device)
                done = torch.zeros(B, dtype=torch.bool, device=device)

                # Precompute allowed-vocab mask once per batch
                # Try to read vocab size from the model; fall back to tokenizer if needed
                try:
                    vocab_size = transformer.base_model.unembed.weight.size(0)
                except AttributeError:
                    vocab_size = len(self.index_to_token)

                allowed = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                allowed[first_codon_id:] = True   # codons (contiguous assumption)
                allowed[end_id] = True            # allow END as well

                # Reasonable cap: AA length (padded) + 1 for END, bounded by max_seq_length
                cap = min(max_seq_length, int(batch_aa.size(1)) + 2)

                for _ in range(cap):
                    input_ids = torch.cat([prefix, generated], dim=1)

                    # mask: 1.0 keep, 0.0 pad (float/bf16 to match model math)
                    attn = (input_ids != pad_id).to(
                        dtype=transformer.base_model.embedding_layer.weight.dtype,
                        device=input_ids.device
                    )

                    out = transformer(input_ids, attention_mask=attn)
                    logits = out.logits[:, -1, :]  # (B, V)

                    # Restrict to codons + END
                    masked = logits.clone()
                    masked[:, ~allowed] = -1e9

                    if temperature <= 0:
                        next_tokens = masked.argmax(dim=-1, keepdim=True)
                    else:
                        probs = torch.softmax(masked / max(1e-8, temperature), dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=1)

                    next_flat = next_tokens.squeeze(1)
                    generated = torch.cat([generated, next_tokens], dim=1)

                    # mark finished
                    just_finished = (next_flat == end_id) & (~done)
                    done |= just_finished
                    if done.all():
                        break

                # Convert this batch back to strings
                gen_np = generated.cpu().numpy()  # NOTE: no [:,1:] here
                if return_string:
                    batch_nt = self._detokenise_nt_seqs(gen_np)
                    outputs.extend(batch_nt)
                else:
                    outputs.append(gen_np)

        if not return_string:
            outputs = np.vstack(outputs)

        return np.array(outputs)



    def _detokenise_nt_seqs(self, nt_token_array):
        """
        Converts a batch of tokenised codon sequences (as NumPy array) back into nucleotide strings.
        """
        nt_seqs = []
        for seq in nt_token_array:
            codons = []
            for token_id in seq:
                if token_id == self.end_token_id:
                    break
                if token_id in [self.start_token_id, self.pad_token_id]:
                    continue
                codon = self.index_to_token.get(token_id, 'NNN')
                codons.append(codon)
            nt_seqs.append("".join(codons))
        return nt_seqs



    def _detokenise_aa_seqs(self, aa_id_seqs):
        """
        Convert a list of amino acid token ID sequences into human-readable strings,
        removing special tokens like <PAD>, <SEP>, <START>, and <END>.

        Args:
            aa_id_seqs (List[List[int]]): A list where each element is a list of amino acid token IDs.

        Returns:
            List[str]: Decoded amino acid sequences as strings.
        """
        aa_seqs = []

        # Define set of special tokens to ignore
        special_token_ids = {
            self.pad_token_id,
            self.sep_token_id,
            self.start_token_id,
            self.end_token_id
        }

        for id_seq in aa_id_seqs:
            # Convert to NumPy array if it's a PyTorch tensor
            if isinstance(id_seq, torch.Tensor):
                id_seq = id_seq.cpu().numpy()

            aa_chars = []
            for token_id in id_seq:
                if token_id in special_token_ids:
                    continue  # skip special tokens entirely
                aa = self.index_to_token.get(token_id, '')  # fallback to '' if unknown
                aa_chars.append(aa)

            aa_seq = ''.join(aa_chars)
            aa_seqs.append(aa_seq)

        return aa_seqs


    def compute_levenshtein_distances(self, predictions, references, normalize=True):
        """
        Compute Levenshtein distances (raw or normalised) between predicted and reference sequences.

        Args:
            predictions (list[str]): List of predicted nucleotide sequences.
            references (list[str]): List of ground truth nucleotide sequences.
            normalize (bool): If True, return normalised distances (0–1). Else, return raw distances.

        Returns:
            list[float]: List of distances (float if normalised, int otherwise)
        """
        distances = []
        for pred, ref in zip(predictions, references):
            if len(ref) == 0:
                distances.append(1.0 if normalize else len(pred))  # Fallback for empty reference
                continue

            dist = Levenshtein.distance(pred, ref)
            if normalize:
                dist = dist / max(len(pred), len(ref))

            distances.append(dist)

        return distances

