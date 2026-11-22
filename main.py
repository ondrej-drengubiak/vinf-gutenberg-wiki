import importlib
import os
import sys
import unittest

from config import Config
from crawler import Crawler
from fetcher import Fetcher
from extractor import Extractor
from storage import Storage
from indexer import Indexer
from searcher import Searcher
import test_extractor

# Add the spark/ and index2/ folders to the path so their modules can be imported.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE_DIR, 'spark'))
sys.path.insert(0, os.path.join(_BASE_DIR, 'index2'))

WIDTH = 70
_BACK = object()   # sentinel: user chose "0. Back" in a nested question


# ---------------------------------------------------------------------------
# Unified menu rendering — every menu in the app uses these helpers.
# ---------------------------------------------------------------------------

def line(ch='='):
    print(ch * WIDTH)


def print_header(title):
    print()
    line()
    print(f">>> {title}")
    line()


def print_option(key, label, desc=''):
    if desc:
        print(f"{key:>3}. {label:<20}— {desc}")
    else:
        print(f"{key:>3}. {label}")


def print_back():
    print("Back to menu.")


def print_invalid():
    print("Invalid option.")


def ask_menu(title, options, default=None, note=None):
    """
    Render a submenu in the unified format and return the user's choice
    as a string ('0' = Back). options = [(key, label, description), ...]
    """
    print_header(title)
    for key, label, desc in options:
        print_option(key, label, desc)
    print_option(0, 'Back')
    if note:
        print(note)
    line()
    prompt = f"Choose [{default}]: " if default else "Choose: "
    choice = input(prompt).strip()
    return choice or (default or '')


def confirm(prompt):
    return input(prompt).strip().lower() == 'yes'


def print_main_menu():
    print_header("MAIN MENU")
    print("PART 1 — CRAWLER, EXTRACTOR, INDEXER, SEARCHER")
    line('-')
    print_option(1, 'Crawl', 'download HTML pages + extract links')
    print_option(2, 'Fetch Text', 'download plain text for crawled books')
    print_option(3, 'Extract', 'extract metadata from HTML to TSV')
    print_option(4, 'Index', 'build inverted index (metadata / text / both)')
    print_option(5, 'Delete TSV', 'delete gutenberg.tsv (HTML/TXT are kept!)')
    print_option(6, 'Delete Index', 'delete index file(s) (TSV/HTML/TXT are kept!)')
    print_option(7, 'Part 1 Statistics', 'crawl + both indexes')
    print_option(8, 'Search', 'query the index (metadata / text)')
    print_option(9, 'Run Tests', 'run the REGEX extractor unit tests')
    line()
    print("PART 2 — SPARK, WIKIPEDIA, JOIN")
    line('-')
    print_option(10, 'Spark Extractor', 'Gutenberg HTMLs -> Parquet')
    print_option(11, 'Wiki Parser', 'Wikipedia dump -> Parquet (only matching titles)')
    print_option(12, 'Join', 'Gutenberg + Wiki -> Parquet')
    print_option(13, 'Preview Parquet', 'show rows from Parquet files')
    print_option(14, 'Part 2 Statistics', 'Parquet sizes, wiki match, join coverage')
    line()
    print("PART 3 — OWN FIELDED INDEX (Lucene-style, pure Python)")
    line('-')
    print_option(15, 'Spark Export', 'joined.parquet -> JSON lines')
    print_option(16, 'Build Fielded Index', 'fields: title, author, subjects, abstract, ...')
    print_option(17, 'Search (fielded)', 'term / phrase / fuzzy / range / boolean')
    print_option(18, 'Compare Old vs New', 'text report for the wiki')
    print_option(19, 'Fielded Statistics', 'vocabulary, postings, fields')
    line()
    print_option(0, 'Exit')
    line()


# ---------------------------------------------------------------------------
# PART 1
# ---------------------------------------------------------------------------

def _build_index(config, index_type, title):
    print_header(f"INDEX: {title}")
    indexer = Indexer(config, index_type=index_type)
    try:
        indexer.build_index()
    except KeyboardInterrupt:
        print("\n\nSTOPPED BY USER (Ctrl+C)")
        print(f"{title.capitalize()} index state saved. Run again to continue.")
    indexer.print_stats()


def submenu_index(config):
    sub = ask_menu("INDEX — WHAT TO INDEX?", [
        (1, 'Metadata only', 'title, author, subjects, loc_class'),
        (2, 'Full text only', 'gutenberg_data/<id>.txt'),
        (3, 'Both', 'metadata + full text, two separate indexes'),
    ])

    if sub == '1':
        _build_index(config, 'metadata', 'METADATA')
    elif sub == '2':
        _build_index(config, 'text', 'FULL TEXT')
    elif sub == '3':
        _build_index(config, 'metadata', 'METADATA')
        _build_index(config, 'text', 'FULL TEXT')
    elif sub == '0':
        print_back()
    else:
        print_invalid()


def submenu_search(config):
    sub = ask_menu("SEARCH — WHICH INDEX?", [
        (1, 'Metadata index', ''),
        (2, 'Full text index', ''),
    ])
    if sub == '1':
        index_type = 'metadata'
    elif sub == '2':
        index_type = 'text'
    elif sub == '0':
        print_back()
        return
    else:
        print_invalid()
        return

    method_choice = ask_menu("SEARCH — WEIGHTING METHOD", [
        (1, 'tfidf', 'classic: tf * log10(N/df)'),
        (2, 'sublinear', '(1+log10(tf)) * log10(1 + N/df)'),
        (3, 'both', 'side by side'),
    ])
    if method_choice == '0':
        print_back()
        return
    if method_choice not in ('1', '2', '3'):
        print_invalid()
        return

    try:
        searcher = Searcher(config, index_type=index_type)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    print_header(f"SEARCH — {index_type.upper()} INDEX")
    while True:
        query = input("\nQuery (empty to go back): ").strip()
        if not query:
            print_back()
            break

        try:
            top_k = int(input("Top-K [10]: ").strip() or "10")
        except ValueError:
            top_k = 10

        if method_choice == '1':
            results = searcher.search(query, method='tfidf', top_k=top_k)
            searcher.print_results(query, results, 'tfidf', top_k)
        elif method_choice == '2':
            results = searcher.search(query, method='sublinear', top_k=top_k)
            searcher.print_results(query, results, 'sublinear', top_k)
        else:
            r1 = searcher.search(query, method='tfidf', top_k=top_k)
            r2 = searcher.search(query, method='sublinear', top_k=top_k)
            searcher.print_results(query, r1, 'tfidf', top_k)
            searcher.print_results(query, r2, 'sublinear', top_k)


def submenu_delete_index(config):
    sub = ask_menu("DELETE INDEX — WHAT TO DELETE?", [
        (1, 'Metadata index', 'metadata index only'),
        (2, 'Full text index', 'full text index only'),
        (3, 'Both indexes', ''),
    ], note="TSV, HTML and TXT files are NOT touched.")

    if sub == '1':
        if confirm("Type 'yes' to delete METADATA index: "):
            Indexer(config, index_type='metadata').delete_index()
        else:
            print("Cancelled.")
    elif sub == '2':
        if confirm("Type 'yes' to delete TEXT index: "):
            Indexer(config, index_type='text').delete_index()
        else:
            print("Cancelled.")
    elif sub == '3':
        if confirm("Type 'yes' to delete BOTH indexes: "):
            Indexer(config, index_type='metadata').delete_index()
            Indexer(config, index_type='text').delete_index()
        else:
            print("Cancelled.")
    elif sub == '0':
        print_back()
    else:
        print_invalid()


def _ask_text_sample_limit():
    sub = ask_menu("TEXT TOKEN COUNT — SAMPLE?", [
        (1, 'Estimate', 'from first 500 files (fast, recommended)'),
        (2, 'Exact', 'count over ALL files (hours)'),
    ], default='1', note="Text token counting over 28 GB is slow.")
    if sub == '1':
        return 500
    if sub == '2':
        return None
    if sub == '0':
        return _BACK
    print("Invalid option, using 500-file estimate.")
    return 500


def _ask_token_mode():
    """Returns (mode, sample_limit) or _BACK."""
    sub = ask_menu("COUNT TOKENS (tiktoken)?", [
        (1, 'No', 'skip token counting (fastest)'),
        (2, 'Metadata only', 'TSV title/author/subjects/loc_class'),
        (3, 'Text only', '.txt files (28 GB, slow unless sampled)'),
        (4, 'Both', 'metadata + text'),
    ], default='1')

    if sub == '0':
        return _BACK
    if sub == '1':
        return 'none', None
    if sub == '2':
        return 'metadata', None
    if sub in ('3', '4'):
        limit = _ask_text_sample_limit()
        if limit is _BACK:
            return _BACK
        return ('text' if sub == '3' else 'both'), limit
    print("Invalid option, skipping token count.")
    return 'none', None


def submenu_stats(config, storage):
    answer = _ask_token_mode()
    if answer is _BACK:
        print_back()
        return
    token_mode, token_sample_limit = answer

    print_header("PART 1 STATISTICS — CRAWL")
    storage.print_stats(token_mode=token_mode, token_sample_limit=token_sample_limit)

    print_header("PART 1 STATISTICS — METADATA INDEX")
    indexer_meta = Indexer(config, index_type='metadata')
    if indexer_meta.indexed_docs:
        indexer_meta.print_stats()
    else:
        print("  No metadata index found. Run Index (4) -> Metadata only first.")

    print_header("PART 1 STATISTICS — FULL TEXT INDEX")
    indexer_text = Indexer(config, index_type='text')
    if indexer_text.indexed_docs:
        indexer_text.print_stats()
    else:
        print("  No text index found. Run Index (4) -> Full text only first.")


def run_tests():
    print_header("RUN TESTS — test_extractor.py")

    importlib.reload(test_extractor)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_extractor)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    line()
    print(f"TESTS FINISHED: {result.testsRun} run, "
          f"{len(result.failures)} failed, "
          f"{len(result.errors)} errors, "
          f"{len(result.skipped)} skipped")
    line()


# ---------------------------------------------------------------------------
# PART 2
# ---------------------------------------------------------------------------

def run_spark_job(title, module_name, warning=None):
    print_header(title)
    if warning:
        print(f"WARNING: {warning}")
    try:
        module = importlib.import_module(module_name)
        module.main()
    except KeyboardInterrupt:
        print("\n\nSTOPPED BY USER (Ctrl+C)")
        print("Spark job interrupted. Run it again to restart.")
    except Exception as e:
        print(f"ERROR: {e}")


def submenu_preview():
    sub = ask_menu("PREVIEW PARQUET — WHICH FILE?", [
        (1, 'Gutenberg books', 'spark_output/gutenberg_books.parquet'),
        (2, 'Wikipedia pages', 'spark_output/wiki_pages.parquet'),
        (3, 'Joined', 'spark_output/joined.parquet'),
    ])
    names = {'1': 'gutenberg', '2': 'wiki', '3': 'joined'}

    if sub == '0':
        print_back()
        return
    if sub not in names:
        print_invalid()
        return

    try:
        n = int(input("Rows [5]: ").strip() or "5")
    except ValueError:
        n = 5

    try:
        import spark_stats
        spark_stats.preview(names[sub], n)
    except Exception as e:
        print(f"ERROR: {e}")


def spark_statistics():
    print_header("PART 2 STATISTICS")
    try:
        import spark_stats
        spark_stats.show_stats()
    except KeyboardInterrupt:
        print("\n\nSTOPPED BY USER (Ctrl+C)")
    except Exception as e:
        print(f"ERROR: {e}")


# ---------------------------------------------------------------------------
# PART 3 — own fielded index (index2/)
# ---------------------------------------------------------------------------

def run_module(title, module_name, func_name='main'):
    print_header(title)
    try:
        module = importlib.import_module(module_name)
        getattr(module, func_name)()
    except KeyboardInterrupt:
        print("\n\nSTOPPED BY USER (Ctrl+C)")
    except Exception as e:
        print(f"ERROR: {e}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    config = Config()
    storage = Storage(config)

    while True:
        print_main_menu()
        choice = input("Choose: ").strip()

        # ----- PART 1 -----
        if choice == '1':
            print_header("CRAWL")
            crawler = Crawler(config)
            try:
                crawler.crawl()
            except KeyboardInterrupt:
                print("\n\nSTOPPED BY USER (Ctrl+C)")
                print("Crawl state saved. Run again to continue.")
            storage.print_stats()

        elif choice == '2':
            print_header("FETCH TEXT")
            fetcher = Fetcher(config)
            try:
                fetcher.fetch_all_texts()
            except KeyboardInterrupt:
                print("\n\nSTOPPED BY USER (Ctrl+C)")
                print("Already fetched texts are saved. Run again to continue.")
            storage.print_stats()

        elif choice == '3':
            print_header("EXTRACT")
            extractor = Extractor(config)
            try:
                extractor.extract_all()
            except KeyboardInterrupt:
                print("\n\nSTOPPED BY USER (Ctrl+C)")
                print("Already extracted records are saved. Run again to continue.")
            storage.print_stats()

        elif choice == '4':
            submenu_index(config)

        elif choice == '5':
            print_header("DELETE TSV")
            print(f"This will permanently delete: {config.TSV_FILE}")
            print("Crawled HTML and TXT files in", config.OUTPUT_DIR, "are NOT touched.")
            if confirm("Type 'yes' to confirm: "):
                Extractor(config).delete_tsv()
            else:
                print("Cancelled.")
            storage.print_stats()

        elif choice == '6':
            submenu_delete_index(config)

        elif choice == '7':
            submenu_stats(config, storage)

        elif choice == '8':
            submenu_search(config)

        elif choice == '9':
            run_tests()

        # ----- PART 2 -----
        elif choice == '10':
            run_spark_job("SPARK EXTRACTOR", "spark_extractor")

        elif choice == '11':
            run_spark_job("WIKI PARSER", "spark_wiki_parser",
                          warning="full EN dump (~110 GB) takes a long time. "
                                  "Do not close the terminal.")

        elif choice == '12':
            run_spark_job("JOIN", "spark_join")

        elif choice == '13':
            submenu_preview()

        elif choice == '14':
            spark_statistics()

        # ----- PART 3 -----
        elif choice == '15':
            run_spark_job("SPARK EXPORT", "spark_export")

        elif choice == '16':
            run_module("BUILD FIELDED INDEX", "index_builder")

        elif choice == '17':
            run_module("SEARCH — FIELDED INDEX", "index_search", "interactive")

        elif choice == '18':
            run_module("COMPARE OLD vs NEW INDEX", "compare_indexes")

        elif choice == '19':
            run_module("FIELDED INDEX STATISTICS", "index_search", "show_stats")

        # ----- EXIT -----
        elif choice == '0':
            print("Bye.")
            sys.exit(0)

        else:
            print_invalid()


if __name__ == '__main__':
    main()