"""

storage.py

"""
import os
import csv

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


class Storage:
    """Helpers for storage and statistics."""

    
    def __init__(self, config):
        self.config = config
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self._encoding = tiktoken.get_encoding("cl100k_base") if _TIKTOKEN_AVAILABLE else None

    def count_html(self):
        """Count HTML files in output dir."""
        if not os.path.exists(self.config.OUTPUT_DIR):
            return 0
        return sum(1 for f in os.listdir(self.config.OUTPUT_DIR) if f.endswith('.html'))

    def count_txt(self):
        """Count TXT files in output dir."""
        if not os.path.exists(self.config.OUTPUT_DIR):
            return 0
        return sum(1 for f in os.listdir(self.config.OUTPUT_DIR) if f.endswith('.txt'))

    def count_tsv(self):
        """Count records in TSV."""
        if not os.path.exists(self.config.TSV_FILE):
            return 0
        with open(self.config.TSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            return max(0, sum(1 for _ in reader) - 1)  # minus header

    def total_html_size(self):
        if not os.path.exists(self.config.OUTPUT_DIR):
            return 0
        total = 0
        for f in os.listdir(self.config.OUTPUT_DIR):
            if f.endswith('.html'):
                total += os.path.getsize(os.path.join(self.config.OUTPUT_DIR, f))
        return total

    def total_txt_size(self):
        if not os.path.exists(self.config.OUTPUT_DIR):
            return 0
        total = 0
        for f in os.listdir(self.config.OUTPUT_DIR):
            if f.endswith('.txt'):
                total += os.path.getsize(os.path.join(self.config.OUTPUT_DIR, f))
        return total

    def total_tokens_metadata(self):
        if not _TIKTOKEN_AVAILABLE:
            return None, False
        if not os.path.exists(self.config.TSV_FILE):
            return 0, False

        fields = self.config.INDEX_METADATA_FIELDS
        total = 0
        with open(self.config.TSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                for field in fields:
                    value = row.get(field, '') or ''
                    if value:
                        total += len(self._encoding.encode(value, disallowed_special=()))
        return total, False

    def total_tokens_text(self, sample_limit=None):
        if not _TIKTOKEN_AVAILABLE:
            return None, False
        if not os.path.exists(self.config.OUTPUT_DIR):
            return 0, False

        txt_files = sorted(
            f for f in os.listdir(self.config.OUTPUT_DIR) if f.endswith('.txt')
        )
        if not txt_files:
            return 0, False

        estimated = sample_limit is not None and sample_limit < len(txt_files)
        files_to_read = txt_files[:sample_limit] if estimated else txt_files

        total = 0
        for filename in files_to_read:
            path = os.path.join(self.config.OUTPUT_DIR, filename)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            total += len(self._encoding.encode(text, disallowed_special=()))

        if estimated:
            total = int(total * (len(txt_files) / len(files_to_read)))

        return total, estimated

    def _print_token_line(self, label, tokens, estimated):
        if tokens is None:
            print(f"  {label}: tiktoken not installed "
                  f"(pip install tiktoken --break-system-packages)")
        else:
            suffix = " (estimated from sample)" if estimated else ""
            print(f"  {label}: {tokens:,}{suffix}")

    def print_stats(self, token_mode='none', token_sample_limit=None):
        print()
        print("=" * 70)
        print("STATISTICS")
        print("=" * 70)
        print(f"  HTML files:      {self.count_html()}")
        print(f"  TXT files:       {self.count_txt()}")
        print(f"  TSV records:     {self.count_tsv()}")
        print(f"  Total HTML size: {self.total_html_size()/1024/1024:.2f} MB")
        print(f"  Total text size: {self.total_txt_size()/1024/1024/1024:.2f} GB")

        if token_mode in ('metadata', 'both'):
            tokens, est = self.total_tokens_metadata()
            self._print_token_line("Metadata tokens", tokens, est)

        if token_mode in ('text', 'both'):
            tokens, est = self.total_tokens_text(sample_limit=token_sample_limit)
            self._print_token_line("Text tokens", tokens, est)

        if token_mode == 'none':
            print("  Total tokens:    (skipped)")

        print("=" * 70)