#!/usr/bin/env python3
"""
AlgoStock CLI - Unified command-line interface for KRX stock data management and analysis.

Combines ETL operations and stock screening into a single tool with subcommands:
    algostock etl backfill    - Run historical data backfill
    algostock etl update      - Run daily update or catch-up
    algostock etl status      - Show database status
    algostock etl validate    - Validate data integrity
    algostock etl cleanup     - Clean up old data

    algostock analyze price   - Screen by price increase
    algostock analyze value   - Screen by trading value
    algostock analyze combined - Combined screening

    algostock quick           - Quick screening for current year

    algostock adj-price build     - Build adj_daily_prices from change_rate chain (수정주가)
    algostock adj-price validate  - Validate Samsung 2018 50:1 split spot-check
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta

# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def validate_date(date_str: str) -> str:
    """Validate date format (YYYYMMDD)."""
    try:
        datetime.strptime(date_str, '%Y%m%d')
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Expected YYYYMMDD")


def validate_percentile(percentile: str) -> int:
    """Validate percentile value (1-100)."""
    try:
        value = int(percentile)
        if 1 <= value <= 100:
            return value
        raise argparse.ArgumentTypeError(f"Percentile must be between 1 and 100, got {value}")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid percentile: {percentile}. Must be an integer")


def validate_markets(markets_str: str) -> list:
    """Validate markets list."""
    valid_markets = ['kospi', 'kosdaq', 'kodex']
    markets = [m.strip().lower() for m in markets_str.split(',')]

    for market in markets:
        if market not in valid_markets:
            raise argparse.ArgumentTypeError(
                f"Invalid market: {market}. Valid markets: {', '.join(valid_markets)}"
            )
    return markets


# =============================================================================
# ETL Commands
# =============================================================================

def cmd_etl_status(args):
    """Show ETL status."""
    from etl.clean_etl import CleanETLPipeline

    pipeline = CleanETLPipeline(args.db_path)
    status = pipeline.get_status()

    print("\n=== Database Status ===")
    print(f"Stocks: {status['stocks']:,}")
    print(f"Daily Prices: {status['daily_prices']:,}")
    if status['date_range'][0] and status['date_range'][1]:
        print(f"Date Range: {status['date_range'][0]} to {status['date_range'][1]}")
    else:
        print("Date Range: No data")
    print(f"Total Records: {status['total_records']:,}")


def cmd_etl_validate(args):
    """Validate data integrity."""
    from etl.clean_etl import CleanETLPipeline

    pipeline = CleanETLPipeline(args.db_path)
    validation = pipeline.validate_data()

    print("\n=== Validation Results ===")
    print(f"Orphaned Prices: {validation['orphaned_prices']}")
    print(f"Duplicate Prices: {validation['duplicate_prices']}")
    print(f"Validation Passed: {'Yes' if validation['validation_passed'] else 'No'}")

    if not validation['validation_passed']:
        sys.exit(1)


def cmd_etl_cleanup(args):
    """Clean up old data."""
    from etl.clean_etl import CleanETLPipeline

    logger = logging.getLogger(__name__)
    logger.info(f"Cleaning up data older than {args.days} days...")

    pipeline = CleanETLPipeline(args.db_path)
    pipeline.cleanup_old_data(days_to_keep=args.days)

    logger.info("Cleanup completed!")


def cmd_etl_verify(args):
    """Verify backfill completeness - check for gaps in data."""
    import sqlite3

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    print("\n" + "=" * 70)
    print("BACKFILL VERIFICATION REPORT")
    print("=" * 70)

    # Get date range
    cursor.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM daily_prices")
    min_date, max_date, total_days = cursor.fetchone()

    print(f"\nDate Range: {min_date} to {max_date}")
    print(f"Total Trading Days in DB: {total_days:,}")

    # Calculate expected trading days (approximate: ~250 per year)
    from datetime import datetime
    start_dt = datetime.strptime(min_date, '%Y%m%d')
    end_dt = datetime.strptime(max_date, '%Y%m%d')
    years = (end_dt - start_dt).days / 365.25
    expected_days = int(years * 250)
    print(f"Expected Trading Days (~250/year): ~{expected_days:,}")

    # Check market coverage by year
    print("\n" + "-" * 70)
    print("MARKET COVERAGE BY YEAR")
    print("-" * 70)
    cursor.execute("""
        SELECT SUBSTR(date, 1, 4) as year,
               SUM(CASE WHEN market_type = 'kospi' THEN 1 ELSE 0 END) as kospi,
               SUM(CASE WHEN market_type = 'kosdaq' THEN 1 ELSE 0 END) as kosdaq,
               SUM(CASE WHEN market_type = 'kodex' THEN 1 ELSE 0 END) as kodex,
               COUNT(DISTINCT date) as trading_days
        FROM daily_prices
        GROUP BY year
        ORDER BY year
    """)

    print(f"{'Year':<6} {'KOSPI':>10} {'KOSDAQ':>10} {'KODEX':>10} {'Days':>8} {'Status':<15}")
    print("-" * 70)

    missing_markets = []
    for row in cursor.fetchall():
        year, kospi, kosdaq, kodex, days = row
        status = "OK"
        missing = []
        if kospi == 0:
            missing.append('kospi')
        if kosdaq == 0:
            missing.append('kosdaq')
        if kodex == 0:
            missing.append('kodex')

        if missing:
            status = f"MISSING: {','.join(missing)}"
            missing_markets.append((year, missing))

        print(f"{year:<6} {kospi:>10,} {kosdaq:>10,} {kodex:>10,} {days:>8} {status:<15}")

    # Check for anomalies in daily record counts
    print("\n" + "-" * 70)
    print("DAILY RECORD COUNT ANOMALIES (< 1000 records)")
    print("-" * 70)

    cursor.execute("""
        SELECT date, COUNT(*) as cnt,
               SUM(CASE WHEN market_type = 'kospi' THEN 1 ELSE 0 END) as kospi,
               SUM(CASE WHEN market_type = 'kosdaq' THEN 1 ELSE 0 END) as kosdaq,
               SUM(CASE WHEN market_type = 'kodex' THEN 1 ELSE 0 END) as kodex
        FROM daily_prices
        GROUP BY date
        HAVING COUNT(*) < 1000
        ORDER BY date
    """)

    anomalies = cursor.fetchall()
    dates_to_fix = []

    # First, collect ALL dates that need fixing
    for date, cnt, kospi, kosdaq, kodex in anomalies:
        missing = []
        if kospi == 0:
            missing.append('kospi')
        if kosdaq == 0:
            missing.append('kosdaq')
        if kodex == 0:
            missing.append('kodex')
        if missing:
            dates_to_fix.append((date, missing))

    if anomalies:
        print(f"Found {len(anomalies)} dates with low record counts:")
        print(f"{'Date':<12} {'Total':>8} {'KOSPI':>8} {'KOSDAQ':>8} {'KODEX':>8} {'Missing'}")
        print("-" * 70)
        # Display only first 30 for readability
        for date, cnt, kospi, kosdaq, kodex in anomalies[:30]:
            missing = []
            if kospi == 0:
                missing.append('kospi')
            if kosdaq == 0:
                missing.append('kosdaq')
            if kodex == 0:
                missing.append('kodex')
            print(f"{date:<12} {cnt:>8,} {kospi:>8,} {kosdaq:>8,} {kodex:>8,} {','.join(missing)}")

        if len(anomalies) > 30:
            print(f"... and {len(anomalies) - 30} more dates")
    else:
        print("No anomalies found (all days have >= 1,000 records)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not missing_markets and not dates_to_fix:
        print("✓ Backfill appears COMPLETE - all markets covered")
    else:
        if missing_markets:
            print(f"⚠ Years with missing markets:")
            for year, markets in missing_markets:
                print(f"   {year}: missing {', '.join(markets)}")

        if dates_to_fix:
            print(f"⚠ {len(dates_to_fix)} dates need fixing")

    conn.close()

    # Fix mode
    if args.fix and dates_to_fix:
        print("\n" + "=" * 70)
        print("FIXING MISSING DATA")
        print("=" * 70)

        from etl.clean_etl import CleanETLPipeline
        from etl.krx_api import KRXAPI
        from config import load_config

        config_dict = load_config()
        api = KRXAPI(config_dict['api']['auth_key'], config_dict.get('api', {}))
        pipeline = CleanETLPipeline(args.db_path)

        # Group dates by missing markets
        fix_summary = {}
        for date, missing in dates_to_fix:
            key = tuple(sorted(missing))
            if key not in fix_summary:
                fix_summary[key] = []
            fix_summary[key].append(date)

        total_fixed = 0
        for markets, dates in fix_summary.items():
            markets_list = list(markets)
            print(f"\nFetching {', '.join(markets_list)} for {len(dates)} dates...")

            for date in dates:
                try:
                    market_data = api.fetch_data_for_date_parallel(date, markets_list, is_backfill=True)

                    raw_data = []
                    for market, records in market_data.items():
                        raw_data.extend(records)

                    if raw_data:
                        result = pipeline.process_data(raw_data)
                        print(f"  {date}: +{result['prices_processed']} records")
                        total_fixed += result['prices_processed']
                    else:
                        print(f"  {date}: no data (holiday?)")

                except Exception as e:
                    print(f"  {date}: error - {e}")

        print(f"\nFixed {total_fixed} total records")

    elif args.fix and not dates_to_fix:
        print("\nNothing to fix!")


# =============================================================================
# Index ETL Commands
# =============================================================================

def validate_index_types(index_types_str: str) -> list:
    """Validate index types list."""
    valid_types = ['kospi_index', 'kosdaq_index', 'bond_index', 'govt_bond', 'derivatives']
    types = [t.strip().lower() for t in index_types_str.split(',')]

    for t in types:
        if t not in valid_types:
            raise argparse.ArgumentTypeError(
                f"Invalid index type: {t}. Valid types: {', '.join(valid_types)}"
            )
    return types


def cmd_adj_price_build(args):
    """Build adj_daily_prices table (수정주가 복원)."""
    from etl.adj_price_etl import AdjPriceETL
    etl = AdjPriceETL(db_path=args.db_path)
    etl.run(skip_validate=args.no_validate)


def cmd_adj_price_validate(args):
    """Validate adj_daily_prices table against known Samsung split."""
    from etl.adj_price_etl import AdjPriceETL
    etl = AdjPriceETL(db_path=args.db_path)
    ok = etl.validate_only()
    if not ok:
        sys.exit(1)


def cmd_index_status(args):
    """Show index data status."""
    from etl.index_etl import IndexETLPipeline

    with IndexETLPipeline(args.db_path) as pipeline:
        pipeline.init_tables()
        stats = pipeline.get_stats()

        print("\n" + "=" * 60)
        print("MARKET INDEX DATA STATUS")
        print("=" * 60)

        for table, info in stats.items():
            print(f"\n{table}:")
            print(f"  Records:      {info['count']:,}")
            print(f"  Unique dates: {info['unique_dates']:,}")
            if info['min_date'] and info['max_date']:
                print(f"  Date range:   {info['min_date']} to {info['max_date']}")
            else:
                print(f"  Date range:   No data")

        # Show sample index names
        print("\n" + "-" * 60)
        kospi_indices = pipeline.get_market_index_names('KOSPI')
        if kospi_indices:
            print(f"KOSPI Indices: {len(kospi_indices)} total")

        kosdaq_indices = pipeline.get_market_index_names('KOSDAQ')
        if kosdaq_indices:
            print(f"KOSDAQ Indices: {len(kosdaq_indices)} total")

        bond_indices = pipeline.get_bond_index_names()
        if bond_indices:
            print(f"Bond Indices: {len(bond_indices)} total")

        deriv_indices = pipeline.get_derivative_index_names()
        if deriv_indices:
            print(f"Derivative Indices: {len(deriv_indices)} total")


def cmd_index_backfill(args):
    """Run index data backfill."""
    from etl.index_etl import IndexETLPipeline
    from etl.krx_api import KRXAPI
    from config import load_config

    config_dict = load_config()
    api = KRXAPI(config_dict['api']['auth_key'], config_dict.get('api', {}))

    print(f"Starting index backfill from {args.start_date} to {args.end_date}")
    print(f"Index types: {', '.join(args.index_types)}")
    if args.force:
        print("Force mode enabled - will reprocess existing data")

    with IndexETLPipeline(args.db_path) as pipeline:
        pipeline.init_tables()

        # Generate trading dates
        from datetime import datetime, timedelta
        start = datetime.strptime(args.start_date, '%Y%m%d')
        end = datetime.strptime(args.end_date, '%Y%m%d')

        dates = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                dates.append(current.strftime('%Y%m%d'))
            current += timedelta(days=1)

        # Get existing dates if not forcing.
        # Check ALL requested tables — a date is "complete" only when every
        # requested table has data for it.  Previously only market_indices was
        # checked, so bond/deriv/govt dates were silently skipped.
        if not args.force:
            _TYPE_TABLE_MAP = {
                'kospi_index': 'index_daily_prices',
                'kosdaq_index': 'index_daily_prices',
                'bond_index': 'bond_index_daily',
                'govt_bond': 'govt_bond_daily',
                'derivatives': 'deriv_index_daily',
            }
            tables_needed = {_TYPE_TABLE_MAP[t] for t in args.index_types if t in _TYPE_TABLE_MAP}
            if tables_needed:
                existing_sets = [
                    pipeline.get_existing_dates(args.start_date, args.end_date, tbl)
                    for tbl in tables_needed
                ]
                fully_existing = existing_sets[0].intersection(*existing_sets[1:])
            else:
                fully_existing = set()
            dates = [d for d in dates if d not in fully_existing]
            print(f"Skipping {len(fully_existing)} dates with complete data in all requested tables")

        if not dates:
            print("No dates to process. Use --force to reprocess existing data.")
            return

        print(f"Processing {len(dates)} trading dates...")

        processed = 0
        total_records = {t: 0 for t in args.index_types}

        for i, date in enumerate(dates):
            try:
                print(f"[{i+1}/{len(dates)}] Processing {date}...")
                index_data = api.fetch_index_data_parallel(date, args.index_types, is_backfill=True)
                results = pipeline.process_all_index_data(index_data)

                for idx_type, count in results.items():
                    total_records[idx_type] = total_records.get(idx_type, 0) + count

                processed += 1

            except Exception as e:
                print(f"  Error: {e}")
                continue

        print("\n" + "=" * 60)
        print(f"Backfill completed. Processed {processed} dates.")
        for idx_type, count in total_records.items():
            print(f"  {idx_type}: {count:,} records")


def cmd_index_update(args):
    """Run daily index data update."""
    from etl.index_etl import IndexETLPipeline
    from etl.krx_api import KRXAPI
    from config import load_config

    config_dict = load_config()
    api = KRXAPI(config_dict['api']['auth_key'], config_dict.get('api', {}))

    # Determine date
    if args.date:
        date = args.date
    else:
        from datetime import datetime, timedelta
        yesterday = datetime.now() - timedelta(days=1)
        date = yesterday.strftime('%Y%m%d')

    # Check if weekday
    from datetime import datetime
    dt = datetime.strptime(date, '%Y%m%d')
    if dt.weekday() >= 5:
        print(f"{date} is a weekend, skipping")
        return

    print(f"Running index update for {date}")
    print(f"Index types: {', '.join(args.index_types)}")

    _TYPE_TABLE_MAP = {
        'kospi_index': 'index_daily_prices',
        'kosdaq_index': 'index_daily_prices',
        'bond_index': 'bond_index_daily',
        'govt_bond': 'govt_bond_daily',
        'derivatives': 'deriv_index_daily',
    }

    with IndexETLPipeline(args.db_path) as pipeline:
        pipeline.init_tables()

        # Check if data exists for ALL requested index types (each in its own table).
        # Previously this only checked index_daily_prices (KOSPI/KOSDAQ), causing
        # bond_index_daily / deriv_index_daily / govt_bond_daily to be skipped.
        if not args.force:
            tables_needed = {_TYPE_TABLE_MAP[t] for t in args.index_types if t in _TYPE_TABLE_MAP}
            if tables_needed and all(pipeline.check_date_exists(date, tbl) for tbl in tables_needed):
                print(f"Data already exists for {date} (all requested types). Use --force to reprocess.")
                return

        try:
            index_data = api.fetch_index_data_parallel(date, args.index_types, is_backfill=False)
            results = pipeline.process_all_index_data(index_data)

            print("Processed:")
            for idx_type, count in results.items():
                if count > 0:
                    print(f"  {idx_type}: {count} records")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


def cmd_etl_backfill(args):
    """Run historical data backfill."""
    from etl.clean_etl import CleanETLPipeline
    from etl.krx_api import KRXAPI
    from config import load_config

    pipeline = CleanETLPipeline(args.db_path)

    print(f"Starting backfill from {args.start_date} to {args.end_date}")
    print(f"Markets: {', '.join(args.markets)}")
    if args.force:
        print("Force mode enabled - will reprocess existing data")

    # Load config and initialize API
    config_dict = load_config()
    api = KRXAPI(config_dict['api']['auth_key'], config_dict.get('api', {}))

    # Check for resume capability
    progress_data = pipeline.load_progress()
    start_date_str = args.start_date

    if progress_data and not args.force:
        print(f"Found existing progress: {progress_data.get('progress_percentage', 0):.1f}% complete")
        resume = input("Resume from last processed date? (y/n): ").lower().strip()
        if resume == 'y':
            last_date = progress_data.get('last_processed_date')
            if last_date:
                resume_date = datetime.strptime(last_date, '%Y%m%d') + timedelta(days=1)
                start_date_str = resume_date.strftime('%Y%m%d')
                print(f"Resuming from {start_date_str}")

    # Get progress information
    progress = pipeline.get_backfill_progress(start_date_str, args.end_date)
    print(f"\nTotal trading days: {progress['total_trading_days']}")
    print(f"Already processed: {progress['processed_dates']}")
    print(f"Remaining: {progress['remaining_dates']}")
    print(f"Progress: {progress['progress_percentage']:.1f}%")

    # Determine dates to process
    if args.force:
        dates_to_process = sorted(progress['remaining_dates_list'] + progress['processed_dates_list'])
    else:
        dates_to_process = progress['remaining_dates_list']
        if not dates_to_process:
            print("All dates already processed. Use --force to reprocess.")
            return

    print(f"\nProcessing {len(dates_to_process)} dates...")

    processed_dates = 0
    total_records = 0

    try:
        for date_str in dates_to_process:
            market_data = api.fetch_data_for_date_parallel(date_str, args.markets, is_backfill=True)

            if market_data:
                raw_data = []
                for market, records in market_data.items():
                    raw_data.extend(records)

                if raw_data:
                    result = pipeline.process_data(raw_data)
                    total_records += result['prices_processed']
                    processed_dates += 1
                    print(f"Processed {result['prices_processed']} records for {date_str}")

                    # Save progress every 10 dates
                    if processed_dates % 10 == 0:
                        pipeline.save_progress({
                            'start_date': args.start_date,
                            'end_date': args.end_date,
                            'last_processed_date': date_str,
                            'processed_dates': processed_dates,
                            'total_records': total_records,
                            'markets': args.markets
                        })
            else:
                print(f"No data for {date_str}")

        print(f"\nBackfill completed: {processed_dates} dates, {total_records} records")

        # Clean up progress file if done
        final_progress = pipeline.get_backfill_progress(args.start_date, args.end_date)
        if final_progress['remaining_dates'] == 0:
            import os
            progress_file = f"{args.db_path}.progress.json"
            if os.path.exists(progress_file):
                os.remove(progress_file)

    except Exception as e:
        print(f"Error during backfill: {e}")
        pipeline.save_progress({
            'start_date': args.start_date,
            'end_date': args.end_date,
            'last_processed_date': date_str if 'date_str' in locals() else None,
            'processed_dates': processed_dates,
            'total_records': total_records,
            'markets': args.markets,
            'error': str(e)
        })
        print("Progress saved - you can resume later")
        sys.exit(1)


def cmd_etl_update(args):
    """Run daily update or catch-up from last date."""
    from etl.clean_etl import CleanETLPipeline
    from etl.krx_api import KRXAPI
    from config import load_config

    pipeline = CleanETLPipeline(args.db_path)
    config_dict = load_config()
    api = KRXAPI(config_dict['api']['auth_key'], config_dict.get('api', {}))

    yesterday = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y%m%d')

    # Determine dates to process
    if args.catchup:
        # Catch-up mode: find last date in DB and fill gaps to yesterday
        status = pipeline.get_status()
        if status['date_range'][1]:
            last_date_str = status['date_range'][1]
            last_date = datetime.strptime(last_date_str, '%Y%m%d')
            start_date = last_date + timedelta(days=1)
        else:
            # No data in DB, start from 30 days ago
            start_date = yesterday - timedelta(days=30)
            last_date_str = "none"

        if start_date > yesterday:
            print(f"Database is up to date (last: {last_date_str})")
            return

        print(f"Catch-up mode: {start_date.strftime('%Y%m%d')} → {yesterday_str}")
        print(f"Last date in DB: {last_date_str}")

        # Generate list of dates to process (skip weekends)
        dates_to_process = []
        current = start_date
        while current <= yesterday:
            if current.weekday() < 5:  # Skip weekends
                dates_to_process.append(current.strftime('%Y%m%d'))
            current += timedelta(days=1)

        if not dates_to_process:
            print("No trading days to process")
            return

        print(f"Processing {len(dates_to_process)} trading days...")
        print(f"Markets: {', '.join(args.markets)}")

    elif args.date:
        # Specific date mode
        dates_to_process = [args.date]
        print(f"Running update for {args.date}")
        print(f"Markets: {', '.join(args.markets)}")

    else:
        # Default: yesterday only
        dates_to_process = [yesterday_str]
        print(f"Running daily update for {yesterday_str}")
        print(f"Markets: {', '.join(args.markets)}")

    # Process each date
    total_records = 0
    processed_days = 0

    try:
        for date_str in dates_to_process:
            # Check if data exists (skip unless force)
            if not args.force and pipeline.check_date_exists(date_str):
                print(f"Skipping {date_str} (exists)")
                continue

            market_data = api.fetch_data_for_date_parallel(
                date_str, args.markets, is_backfill=args.catchup
            )

            raw_data = []
            for market, records in market_data.items():
                raw_data.extend(records)

            if raw_data:
                result = pipeline.process_data(raw_data)
                total_records += result['prices_processed']
                processed_days += 1
                print(f"Processed {result['prices_processed']} records for {date_str}")
            else:
                print(f"No data for {date_str} (holiday?)")

        if args.catchup:
            print(f"\nCatch-up complete: {processed_days} days, {total_records} records")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# =============================================================================
# Analysis Commands
# =============================================================================

def cmd_analyze_price(args):
    """Screen by price increase."""
    from analyzer.screeners import StockScreener

    screener = StockScreener(args.db_path)

    print(f"Screening for top {args.percentile}% by price increase...")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Markets: {', '.join(args.markets)}")

    results = screener.top_price_increase(
        args.start_date, args.end_date, args.markets, args.percentile
    )

    if len(results) == 0:
        print("No results found.")
        return

    print(f"\nFound {len(results)} stocks:")
    print(results.to_string())

    if args.export:
        filename = screener.export_results(results, "price_screening")
        print(f"\nExported to: {filename}")


def cmd_analyze_value(args):
    """Screen by trading value."""
    from analyzer.screeners import StockScreener

    screener = StockScreener(args.db_path)

    print(f"Screening for top {args.percentile}% by trading value...")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Markets: {', '.join(args.markets)}")

    results = screener.top_value(
        args.start_date, args.end_date, args.markets, args.percentile
    )

    if len(results) == 0:
        print("No results found.")
        return

    print(f"\nFound {len(results)} stocks:")
    print(results.to_string())

    if args.export:
        filename = screener.export_results(results, "value_screening")
        print(f"\nExported to: {filename}")


def cmd_analyze_combined(args):
    """Combined screening by price and value."""
    from analyzer.screeners import StockScreener

    screener = StockScreener(args.db_path)

    print(f"Combined screening: top {args.price_percentile}% price AND top {args.value_percentile}% value")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Markets: {', '.join(args.markets)}")

    results = screener.combined_screen(
        args.start_date, args.end_date, args.markets,
        args.price_percentile, args.value_percentile
    )

    if len(results) == 0:
        print("No results found.")
        return

    print(f"\nFound {len(results)} stocks meeting both criteria:")
    print(results.to_string())

    if args.export:
        filename = screener.export_results(results, "combined_screening")
        print(f"\nExported to: {filename}")


def cmd_quick(args):
    """Quick screening for current year."""
    from analyzer.screeners import StockScreener

    screener = StockScreener(args.db_path)

    current_year = datetime.now().year
    start_date = f"{current_year}0101"
    end_date = f"{current_year}1231"

    print(f"Quick screening for {current_year}")
    print(f"Top {args.percentile}% by both price increase and trading value")
    print(f"Markets: {', '.join(args.markets)}")

    results = screener.combined_screen(
        start_date, end_date, args.markets, args.percentile, args.percentile
    )

    if len(results) == 0:
        print("No results found.")
        return

    print(f"\nFound {len(results)} top performers for {current_year}:")
    print(results.to_string())

    if args.export:
        filename = screener.export_results(results, f"quick_screen_{current_year}")
        print(f"\nExported to: {filename}")


# =============================================================================
# ML Commands
# =============================================================================

def _print_legacy_ml_backtest_deprecated() -> None:
    print("=" * 70)
    print("LEGACY ML BACKTEST IS DEPRECATED")
    print("=" * 70)
    print("ml/backtest.py 경로는 폐기되었습니다.")
    print("통합 엔진을 사용하세요:")
    print("  python3 scripts/run_backtest.py --help")
    print("예시:")
    print("  python3 scripts/run_backtest.py --start 20200101 --end 20251231 --output raw_reaudit")


def cmd_ml_backtest(args):
    """Run ML factor model backtest."""
    _print_legacy_ml_backtest_deprecated()
    sys.exit(2)


def cmd_ml_predict(args):
    """Get current ML stock picks."""
    _print_legacy_ml_backtest_deprecated()
    sys.exit(2)


def cmd_ml_features(args):
    """Show feature importance from trained model."""
    _print_legacy_ml_backtest_deprecated()
    sys.exit(2)


# =============================================================================
# Update-All Command
# =============================================================================

def cmd_update_all(args):
    """Run all ETL updates in sequence: stock prices, index data, adj prices, constituents, delisted."""
    import types

    print("=" * 60)
    print("FULL DATA UPDATE")
    print("=" * 60)

    # Step 1: ETL catch-up (stock prices)
    print("\n[1/5] Stock prices (etl update --catchup)...")
    etl_args = types.SimpleNamespace(
        db_path=args.db_path,
        catchup=True,
        date=None,
        markets=['kospi', 'kosdaq'],
        force=False,
    )
    try:
        cmd_etl_update(etl_args)
    except SystemExit as e:
        print(f"  ETL update failed (exit {e.code})")

    # Step 2: Index update
    print("\n[2/5] Index data (index update)...")
    index_args = types.SimpleNamespace(
        db_path=args.db_path,
        date=None,
        index_types=['kospi_index', 'kosdaq_index', 'bond_index', 'govt_bond', 'derivatives'],
        force=False,
    )
    try:
        cmd_index_update(index_args)
    except SystemExit as e:
        print(f"  Index update failed (exit {e.code})")

    # Step 3: Adj-price rebuild
    print("\n[3/5] Adjusted prices (adj-price build)...")
    adj_args = types.SimpleNamespace(
        db_path=args.db_path,
        no_validate=False,
    )
    try:
        cmd_adj_price_build(adj_args)
    except SystemExit as e:
        print(f"  Adj-price build failed (exit {e.code})")

    # Step 4: Index constituents (auto catch-up from last DB date)
    print("\n[4/5] Index constituents (catch-up from last date)...")
    try:
        from etl.index_constituents_etl import KRXIndexConstituentsDirect
        processor = KRXIndexConstituentsDirect()
        processor.update(strategy='skip')
    except Exception as e:
        print(f"  Index constituents update failed: {e}")

    # Step 5: Delisted stocks (always full refresh)
    print("\n[5/5] Delisted stocks (full refresh)...")
    try:
        from etl.delisted_stocks_etl import (
            create_database_table,
            download_delisted_stocks,
            insert_delisted_stocks_to_db,
        )
        create_database_table(args.db_path)
        df, _ = download_delisted_stocks()
        if df is not None:
            insert_delisted_stocks_to_db(df, args.db_path)
        else:
            print("  Failed to download delisted stocks")
    except Exception as e:
        print(f"  Delisted stocks update failed: {e}")

    print("\n" + "=" * 60)
    print("FULL UPDATE COMPLETE")
    print("=" * 60)


# =============================================================================
# Backtest Commands
# =============================================================================

def cmd_backtest(args):
    """Run backtesting on combined screening strategy."""
    from analyzer.backtester import Backtester

    print(f"Running backtest...")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Stocks: {args.n_stocks}, Hold: {args.hold_months}m, Rebalance: {args.rebalance_months}m")
    print(f"Weighting: {args.weighting}, Sort by: {args.sort_by}")
    print(f"Markets: {', '.join(args.markets)}")
    print()

    with Backtester(args.db_path) as bt:
        results = bt.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            n_stocks=args.n_stocks,
            holding_months=args.hold_months,
            rebalance_months=args.rebalance_months,
            weighting=args.weighting,
            sort_by=args.sort_by,
            price_percentile=args.price_percentile,
            value_percentile=args.value_percentile,
            markets=args.markets
        )

        # Print report
        bt.print_report(results)

        # Export if requested
        if args.export:
            files = bt.export_results(results, "backtest")
            print(f"\nExported to:")
            for name, path in files.items():
                print(f"  {name}: {path}")


# =============================================================================
# Main CLI Setup
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='algostock',
        description='AlgoStock - KRX Stock Data Management and Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All-in-one update
  algostock update-all                              Update everything (prices + index + adj + constituents + delisted)

  # ETL Operations (Stock Data)
  algostock etl status                              Show database status
  algostock etl backfill -s 20200101 -e 20251231    Backfill historical data
  algostock etl update                              Update yesterday's data
  algostock etl update --catchup                    Auto catch-up to yesterday
  algostock etl validate                            Validate data integrity

  # Index Data (KOSPI/KOSDAQ indices, Bonds, Derivatives)
  algostock index status                            Show index data status
  algostock index backfill -s 20200101 -e 20241231  Backfill index data
  algostock index update                            Update yesterday's index data
  algostock index backfill -s 20240101 -e 20240131 -t kospi_index,kosdaq_index

  # Stock Analysis
  algostock analyze price -s 20250101 -e 20251231 -p 1    Top 1% by price
  algostock analyze value -s 20250101 -e 20251231 -p 5    Top 5% by value
  algostock analyze combined -s 20250101 -e 20251231      Combined screening

  # Quick Analysis
  algostock quick -p 5                              Top 5% for current year

  # Backtesting (Simple)
  algostock backtest -s 20150101 -e 20241231 -n 5   Backtest top 5 stocks

  # ML Factor Model
  algostock ml backtest -s 20160101 -e 20251231     ML backtest (walk-forward)
  algostock ml predict -n 20                        Get top 20 stocks by ML score
  algostock ml features                             Show feature importance
        """
    )

    # Global arguments
    parser.add_argument('--db-path', default='krx_stock_data.db', help='Database path')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # -------------------------------------------------------------------------
    # ETL Commands
    # -------------------------------------------------------------------------
    etl_parser = subparsers.add_parser('etl', help='ETL operations')
    etl_subparsers = etl_parser.add_subparsers(dest='etl_command', help='ETL commands')

    # etl status
    etl_status = etl_subparsers.add_parser('status', help='Show database status')
    etl_status.set_defaults(func=cmd_etl_status)

    # etl validate
    etl_validate = etl_subparsers.add_parser('validate', help='Validate data integrity')
    etl_validate.set_defaults(func=cmd_etl_validate)

    # etl cleanup
    etl_cleanup = etl_subparsers.add_parser('cleanup', help='Clean up old data')
    etl_cleanup.add_argument('--days', type=int, default=365, help='Days to keep')
    etl_cleanup.set_defaults(func=cmd_etl_cleanup)

    # etl verify
    etl_verify = etl_subparsers.add_parser('verify', help='Verify backfill completeness')
    etl_verify.add_argument('--fix', action='store_true',
                            help='Automatically fix missing market data')
    etl_verify.set_defaults(func=cmd_etl_verify)

    # etl backfill
    etl_backfill = etl_subparsers.add_parser('backfill', help='Run historical backfill')
    etl_backfill.add_argument('-s', '--start-date', type=validate_date, required=True,
                              help='Start date (YYYYMMDD)')
    etl_backfill.add_argument('-e', '--end-date', type=validate_date, required=True,
                              help='End date (YYYYMMDD)')
    etl_backfill.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                              help='Markets (comma-separated)')
    etl_backfill.add_argument('--force', action='store_true', help='Force reprocess existing data')
    etl_backfill.set_defaults(func=cmd_etl_backfill)

    # etl update
    etl_update = etl_subparsers.add_parser('update', help='Run daily update or catch-up')
    etl_update.add_argument('--date', type=validate_date, help='Specific date (YYYYMMDD)')
    etl_update.add_argument('--catchup', action='store_true',
                            help='Auto catch-up from last DB date to yesterday')
    etl_update.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                            help='Markets (comma-separated)')
    etl_update.add_argument('--force', action='store_true', help='Force reprocess existing data')
    etl_update.set_defaults(func=cmd_etl_update)

    # -------------------------------------------------------------------------
    # Index Commands (KOSPI/KOSDAQ indices, bonds, derivatives)
    # -------------------------------------------------------------------------
    index_parser = subparsers.add_parser('index', help='Market index data operations')
    index_subparsers = index_parser.add_subparsers(dest='index_command', help='Index commands')

    # index status
    index_status = index_subparsers.add_parser('status', help='Show index data status')
    index_status.set_defaults(func=cmd_index_status)

    # index backfill
    index_backfill = index_subparsers.add_parser('backfill', help='Backfill historical index data')
    index_backfill.add_argument('-s', '--start-date', type=validate_date, required=True,
                                help='Start date (YYYYMMDD)')
    index_backfill.add_argument('-e', '--end-date', type=validate_date, required=True,
                                help='End date (YYYYMMDD)')
    index_backfill.add_argument('-t', '--index-types', type=validate_index_types,
                                default='kospi_index,kosdaq_index,bond_index,govt_bond,derivatives',
                                help='Index types (comma-separated): kospi_index,kosdaq_index,bond_index,govt_bond,derivatives')
    index_backfill.add_argument('--force', action='store_true', help='Force reprocess existing data')
    index_backfill.set_defaults(func=cmd_index_backfill)

    # index update
    index_update = index_subparsers.add_parser('update', help='Run daily index update')
    index_update.add_argument('--date', type=validate_date, help='Specific date (YYYYMMDD)')
    index_update.add_argument('-t', '--index-types', type=validate_index_types,
                              default='kospi_index,kosdaq_index,bond_index,govt_bond,derivatives',
                              help='Index types (comma-separated)')
    index_update.add_argument('--force', action='store_true', help='Force reprocess existing data')
    index_update.set_defaults(func=cmd_index_update)

    # -------------------------------------------------------------------------
    # Adjusted Price Commands (수정주가)
    # -------------------------------------------------------------------------
    adj_parser = subparsers.add_parser('adj-price', help='Adjusted price (수정주가) operations')
    adj_subparsers = adj_parser.add_subparsers(dest='adj_command', help='adj-price commands')

    # adj-price build
    adj_build = adj_subparsers.add_parser('build', help='Build adj_daily_prices table from change_rate chain')
    adj_build.add_argument('--no-validate', action='store_true',
                           help='Skip Samsung 2018 split validation after build')
    adj_build.set_defaults(func=cmd_adj_price_build)

    # adj-price validate
    adj_val = adj_subparsers.add_parser('validate', help='Validate adj_daily_prices (Samsung split spot-check)')
    adj_val.set_defaults(func=cmd_adj_price_validate)

    # -------------------------------------------------------------------------
    # Analysis Commands
    # -------------------------------------------------------------------------
    analyze_parser = subparsers.add_parser('analyze', help='Stock analysis')
    analyze_subparsers = analyze_parser.add_subparsers(dest='analyze_command', help='Analysis commands')

    # Common analysis arguments
    def add_analysis_args(p, combined=False):
        p.add_argument('-s', '--start-date', type=validate_date, required=True,
                       help='Start date (YYYYMMDD)')
        p.add_argument('-e', '--end-date', type=validate_date, required=True,
                       help='End date (YYYYMMDD)')
        p.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                       help='Markets (comma-separated)')
        if combined:
            p.add_argument('--price-percentile', type=validate_percentile, default=5,
                           help='Top percentile for price (1-100)')
            p.add_argument('--value-percentile', type=validate_percentile, default=5,
                           help='Top percentile for value (1-100)')
        else:
            p.add_argument('-p', '--percentile', type=validate_percentile, default=1,
                           help='Top percentile (1-100)')
        p.add_argument('--export', action='store_true', help='Export to Excel')

    # analyze price
    analyze_price = analyze_subparsers.add_parser('price', help='Screen by price increase')
    add_analysis_args(analyze_price)
    analyze_price.set_defaults(func=cmd_analyze_price)

    # analyze value
    analyze_value = analyze_subparsers.add_parser('value', help='Screen by trading value')
    add_analysis_args(analyze_value)
    analyze_value.set_defaults(func=cmd_analyze_value)

    # analyze combined
    analyze_combined = analyze_subparsers.add_parser('combined', help='Combined screening')
    add_analysis_args(analyze_combined, combined=True)
    analyze_combined.set_defaults(func=cmd_analyze_combined)

    # -------------------------------------------------------------------------
    # Update-All Command
    # -------------------------------------------------------------------------
    update_all_parser = subparsers.add_parser(
        'update-all',
        help='Run all ETL updates in sequence (stock prices + index + adj prices)',
    )
    update_all_parser.set_defaults(func=cmd_update_all)

    # -------------------------------------------------------------------------
    # Quick Command
    # -------------------------------------------------------------------------
    quick_parser = subparsers.add_parser('quick', help='Quick screening for current year')
    quick_parser.add_argument('-p', '--percentile', type=validate_percentile, default=5,
                              help='Top percentile (1-100)')
    quick_parser.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                              help='Markets (comma-separated)')
    quick_parser.add_argument('--export', action='store_true', help='Export to Excel')
    quick_parser.set_defaults(func=cmd_quick)

    # -------------------------------------------------------------------------
    # Backtest Command
    # -------------------------------------------------------------------------
    backtest_parser = subparsers.add_parser('backtest', help='Backtest screening strategy')
    backtest_parser.add_argument('-s', '--start-date', type=validate_date, required=True,
                                 help='Backtest start date (YYYYMMDD) - first holding period starts here')
    backtest_parser.add_argument('-e', '--end-date', type=validate_date, required=True,
                                 help='Backtest end date (YYYYMMDD)')
    backtest_parser.add_argument('-n', '--n-stocks', type=int, default=5,
                                 help='Number of top stocks to select (default: 5)')
    backtest_parser.add_argument('--hold-months', type=int, default=12,
                                 help='Holding period in months (default: 12)')
    backtest_parser.add_argument('--rebalance-months', type=int, default=12,
                                 help='Rebalancing interval in months (default: 12)')
    backtest_parser.add_argument('-w', '--weighting', default='equal',
                                 choices=['equal', 'value', 'inverse_value', 'mcap', 'inverse_mcap'],
                                 help='Portfolio weighting method (default: equal)')
    backtest_parser.add_argument('--sort-by', default='price',
                                 choices=['price', 'value', 'combined'],
                                 help='How to rank top N: price (return), value (trading value), combined (default: price)')
    backtest_parser.add_argument('--price-percentile', type=validate_percentile, default=5,
                                 help='Top percentile for price screening (default: 5)')
    backtest_parser.add_argument('--value-percentile', type=validate_percentile, default=5,
                                 help='Top percentile for value screening (default: 5)')
    backtest_parser.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                                 help='Markets (comma-separated)')
    backtest_parser.add_argument('--export', action='store_true', help='Export results to CSV')
    backtest_parser.set_defaults(func=cmd_backtest)

    # -------------------------------------------------------------------------
    # ML Commands
    # -------------------------------------------------------------------------
    ml_parser = subparsers.add_parser('ml', help='ML factor ranking model')
    ml_subparsers = ml_parser.add_subparsers(dest='ml_command', help='ML commands')

    # ml backtest
    ml_backtest = ml_subparsers.add_parser('backtest', help='Run ML factor model backtest')
    ml_backtest.add_argument('-s', '--start-date', type=validate_date, required=True,
                             help='Backtest start date (YYYYMMDD)')
    ml_backtest.add_argument('-e', '--end-date', type=validate_date, required=True,
                             help='Backtest end date (YYYYMMDD)')
    ml_backtest.add_argument('-n', '--n-stocks', type=int, default=20,
                             help='Number of stocks to hold (default: 20)')
    ml_backtest.add_argument('--holding-days', type=int, default=21,
                             help='Holding period in days (default: 21 = ~1 month)')
    ml_backtest.add_argument('--train-years', type=int, default=5,
                             help='Years of data for training (default: 5)')
    ml_backtest.add_argument('--target-horizon', type=int, default=21,
                             help='Forward return horizon in days (default: 21)')
    ml_backtest.add_argument('-w', '--weighting', default='equal',
                             choices=['equal', 'score', 'mcap'],
                             help='Portfolio weighting (default: equal)')
    ml_backtest.add_argument('--model-type', default='regressor',
                             choices=['ranker', 'regressor'],
                             help='Model type (default: regressor)')
    ml_backtest.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                             help='Markets (comma-separated)')
    ml_backtest.add_argument('--plot', action='store_true', help='Generate performance charts')
    ml_backtest.add_argument('--export', action='store_true', help='Export results to CSV')
    ml_backtest.set_defaults(func=cmd_ml_backtest)

    # ml predict
    ml_predict = ml_subparsers.add_parser('predict', help='Get current ML stock picks')
    ml_predict.add_argument('-n', '--n-stocks', type=int, default=20,
                            help='Number of stocks to return (default: 20)')
    ml_predict.add_argument('--date', type=validate_date,
                            help='Date for prediction (default: most recent)')
    ml_predict.add_argument('--train-years', type=int, default=5,
                            help='Years of data for training (default: 5)')
    ml_predict.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                            help='Markets (comma-separated)')
    ml_predict.add_argument('--export', action='store_true', help='Export picks to CSV')
    ml_predict.set_defaults(func=cmd_ml_predict)

    # ml features
    ml_features = ml_subparsers.add_parser('features', help='Show feature importance')
    ml_features.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                             help='Markets (comma-separated)')
    ml_features.set_defaults(func=cmd_ml_features)

    # -------------------------------------------------------------------------
    # Parse and Execute
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(args.log_file, log_level)

    # Handle missing commands
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'etl' and not getattr(args, 'etl_command', None):
        etl_parser.print_help()
        sys.exit(1)

    if args.command == 'analyze' and not getattr(args, 'analyze_command', None):
        analyze_parser.print_help()
        sys.exit(1)

    if args.command == 'ml' and not getattr(args, 'ml_command', None):
        ml_parser.print_help()
        sys.exit(1)

    if args.command == 'index' and not getattr(args, 'index_command', None):
        index_parser.print_help()
        sys.exit(1)

    if args.command == 'adj-price' and not getattr(args, 'adj_command', None):
        adj_parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
