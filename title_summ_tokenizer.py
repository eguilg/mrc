from pytorch_pretrained_bert import BertTokenizer, BasicTokenizer


class TSBasicTokenizer(BasicTokenizer):
  def __init__(self, do_lower_case=True):
    super().__init__(do_lower_case)
    self.sep = "||"

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = self._clean_text(text)
    text = self._tokenize_chinese_chars(text)
    orig_tokens = [token for token in text.strip().split(self.sep) if token != '']

    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = split_tokens
    return output_tokens

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(self.sep)
        output.append(char)
        output.append(self.sep)
      else:
        output.append(char)
    return "".join(output)


class TitleSummBertTokenizer(BertTokenizer):
  def __init__(self, vocab_file, do_lower_case=False):
    super().__init__(vocab_file)
    self.basic_tokenizer = TSBasicTokenizer(do_lower_case=do_lower_case)

    self.unused_limit = 300
    self.token_to_unused_map = {}

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      if token == ' ':
        split_tokens.append(token)  # 空格
      else:
        for sub_token in self.wordpiece_tokenizer.tokenize(token):
          split_tokens.append(sub_token)
    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    ids = []
    for token in tokens:
      try:
        ids.append(self.vocab[token])
      except:
        if token not in self.token_to_unused_map:
          if len(self.token_to_unused_map) < self.unused_limit:
            self.token_to_unused_map[token] = '[unused%d]' % (len(self.token_to_unused_map) + 1)
          else:
            raise Exception()
        ids.append(self.vocab[self.token_to_unused_map[token]])
    return ids

  def convert_ids_to_tokens(self, ids):
    """Converts a sequence of ids in wordpiece tokens using the vocab."""
    tokens = []
    for i in ids:
      tokens.append(self.ids_to_tokens[i])
    return tokens
