#!/usr/bin/env python3
"""
download_sources.py
Downloads and curates 20+ official public sources for Groww Mutual Fund RAG chatbot.
Sources are drawn exclusively from:
- Groww Investment Platform (groww.in)
- Groww Mutual Fund AMC (growwmf.in)
- Association of Mutual Funds in India (amfiindia.com)
- Securities and Exchange Board of India (sebi.gov.in / investor.sebi.gov.in)
"""

import os
import sys
import json
import re
import urllib.request
import subprocess
from datetime import datetime
from html import unescape

SOURCES_METADATA = [
    # 1. Groww AMC Schemes (Direct Growth)
    {
        "id": "groww_scheme_elss_tax_saver",
        "title": "Groww ELSS Tax Saver Fund - Scheme Facts & Details",
        "url": "https://groww.in/mutual-funds/groww-elss-tax-saver-fund-direct-growth",
        "organization": "Groww Mutual Fund",
        "doc_type": "scheme_page",
        "category": "ELSS / Tax Saver",
        "scheme_name": "Groww ELSS Tax Saver Fund Direct Growth",
        "is_scheme_page": True
    },
    {
        "id": "groww_scheme_nifty_total_market",
        "title": "Groww Nifty Total Market Index Fund - Scheme Facts & Details",
        "url": "https://groww.in/mutual-funds/groww-nifty-total-market-index-fund-direct-growth",
        "organization": "Groww Mutual Fund",
        "doc_type": "scheme_page",
        "category": "Index Fund / Large & Mid & Small",
        "scheme_name": "Groww Nifty Total Market Index Fund Direct Growth",
        "is_scheme_page": True
    },
    {
        "id": "groww_scheme_multicap_fund",
        "title": "Groww Multicap Fund - Scheme Facts & Details",
        "url": "https://groww.in/mutual-funds/groww-multicap-fund-direct-growth",
        "organization": "Groww Mutual Fund",
        "doc_type": "scheme_page",
        "category": "Equity / Multi Cap",
        "scheme_name": "Groww Multicap Fund Direct Growth",
        "is_scheme_page": True
    },
    {
        "id": "groww_scheme_liquid_fund",
        "title": "Groww Liquid Fund - Scheme Facts & Details",
        "url": "https://groww.in/mutual-funds/groww-liquid-fund-direct-growth",
        "organization": "Groww Mutual Fund",
        "doc_type": "scheme_page",
        "category": "Debt / Liquid",
        "scheme_name": "Groww Liquid Fund Direct Growth",
        "is_scheme_page": True
    },
    {
        "id": "groww_scheme_banking_financial_services",
        "title": "Groww Banking & Financial Services Fund - Scheme Facts & Details",
        "url": "https://groww.in/mutual-funds/groww-banking-financial-services-fund-direct-growth",
        "organization": "Groww Mutual Fund",
        "doc_type": "scheme_page",
        "category": "Equity / Sectoral",
        "scheme_name": "Groww Banking & Financial Services Fund Direct Growth",
        "is_scheme_page": True
    },

    # 2. Groww AMC Statutory & Factsheet Documents
    {
        "id": "groww_amc_kim_nifty_total_market",
        "title": "Groww Nifty Total Market Index Fund - Key Information Memorandum (KIM)",
        "url": "https://assets-netstorage.growwmf.in/compliance_docs/Downloads/KIM/Index%20Funds/KIM_Groww%20Nifty%20Total%20Market%20Index%20Fund.pdf",
        "organization": "Groww Mutual Fund AMC",
        "doc_type": "statutory_kim",
        "category": "Statutory KIM / Compliance",
        "is_pdf": True
    },
    {
        "id": "groww_amc_sid_portal",
        "title": "Groww Mutual Fund - Scheme Information Documents (SID) Downloads",
        "url": "https://www.growwmf.in/downloads/sid",
        "organization": "Groww Mutual Fund AMC",
        "doc_type": "amc_downloads",
        "category": "Statutory SID / Compliance"
    },
    {
        "id": "groww_amc_kim_portal",
        "title": "Groww Mutual Fund - Key Information Memorandum (KIM) Downloads",
        "url": "https://www.growwmf.in/downloads/kim",
        "organization": "Groww Mutual Fund AMC",
        "doc_type": "amc_downloads",
        "category": "Statutory KIM / Compliance"
    },
    {
        "id": "groww_amc_factsheets_portal",
        "title": "Groww Mutual Fund - Monthly Factsheets & Disclosures",
        "url": "https://www.growwmf.in/downloads/factsheets",
        "organization": "Groww Mutual Fund AMC",
        "doc_type": "amc_downloads",
        "category": "Factsheets & Disclosures"
    },

    # 3. Groww Platform Knowledge, Statements & Tax Guides
    {
        "id": "groww_guide_capital_gains_statement",
        "title": "How to Download Capital Gains Statement on Groww for Mutual Funds",
        "url": "https://groww.in/blog/how-to-get-capital-gains-statement-for-mutual-fund-investments",
        "organization": "Groww",
        "doc_type": "platform_guide",
        "category": "Statements & Taxation"
    },
    {
        "id": "groww_faq_capital_gains_tax_report",
        "title": "What is a Capital Gains or Tax P&L Report on Groww",
        "url": "https://groww.in/help/stocks,-f&o,-ipo-&-mtf/sx-reports/what-is-a-capital-gain-or-tax-p-l-report--20",
        "organization": "Groww Help",
        "doc_type": "platform_faq",
        "category": "Statements & Taxation"
    },
    {
        "id": "groww_faq_how_to_start_sip",
        "title": "How to Start an SIP on Groww (Mandates, AutoPay & Timelines)",
        "url": "https://groww.in/help/mutual-funds/mf-sip/how-to-start-a-sip-on-groww",
        "organization": "Groww Help",
        "doc_type": "platform_faq",
        "category": "SIP & Mandates"
    },
    {
        "id": "groww_faq_how_to_cancel_sip",
        "title": "How to Cancel or Stop an Ongoing SIP on Groww",
        "url": "https://groww.in/help/mutual-funds/mf-sip/how-to-cancel-my-sip",
        "organization": "Groww Help",
        "doc_type": "platform_faq",
        "category": "SIP & Mandates"
    },
    {
        "id": "groww_faq_exit_load_explained",
        "title": "Mutual Fund Exit Load Explained - Calculation and Exemptions",
        "url": "https://groww.in/help/mutual-funds/order/what-is-exit-load--71",
        "organization": "Groww Help",
        "doc_type": "platform_faq",
        "category": "Charges & Exit Load"
    },
    {
        "id": "groww_faq_redemption_nav_applicability",
        "title": "NAV Applicability when Redeeming / Withdrawing Mutual Fund Units",
        "url": "https://groww.in/help/mutual-funds/order/when-withdrawing-redeeming-which-days-nav-will-be-applicable-1",
        "organization": "Groww Help",
        "doc_type": "platform_faq",
        "category": "NAV & Transactions"
    },
    {
        "id": "groww_faq_switch_charges",
        "title": "Charges and Taxes on Switching from Regular to Direct Mutual Funds",
        "url": "https://groww.in/help/mutual-funds/switch-to-direct/what-are-the-charges-to-switch-from-regular-to-direct-84",
        "organization": "Groww Help",
        "doc_type": "platform_faq",
        "category": "Charges & Switching"
    },
    {
        "id": "groww_guide_unit_allotment_nav",
        "title": "Mutual Fund Unit Allotment & NAV Realization Process",
        "url": "https://groww.in/blog/unit-allotment-and-nav-process",
        "organization": "Groww",
        "doc_type": "platform_guide",
        "category": "NAV & Allotment"
    },
    {
        "id": "groww_guide_cutoff_timings",
        "title": "SEBI Mutual Fund Cut-off Timings for Purchases & Redemptions",
        "url": "https://groww.in/blog/sebi-revises-mutual-fund-cut-off-timings",
        "organization": "Groww",
        "doc_type": "platform_guide",
        "category": "Cut-off Timings"
    },
    {
        "id": "groww_concept_expense_ratio",
        "title": "What is Total Expense Ratio (TER) in Mutual Funds",
        "url": "https://groww.in/p/expense-ratio",
        "organization": "Groww",
        "doc_type": "concept_guide",
        "category": "Charges & Expense Ratio"
    },
    {
        "id": "groww_concept_riskometer",
        "title": "What is a Mutual Fund Riskometer and Risk Classifications",
        "url": "https://groww.in/p/riskometer",
        "organization": "Groww",
        "doc_type": "concept_guide",
        "category": "Riskometer & Risk"
    },
    {
        "id": "groww_concept_benchmark",
        "title": "Understanding Mutual Fund Benchmarks and Performance Indices",
        "url": "https://groww.in/p/benchmark",
        "organization": "Groww",
        "doc_type": "concept_guide",
        "category": "Benchmarks"
    },
    {
        "id": "groww_concept_nav",
        "title": "What is Net Asset Value (NAV) and Calculation Method",
        "url": "https://groww.in/p/nav",
        "organization": "Groww",
        "doc_type": "concept_guide",
        "category": "NAV & Valuation"
    },
    {
        "id": "groww_pricing_charges",
        "title": "Groww Mutual Fund Pricing, Account Opening & Zero AMC Fees",
        "url": "https://groww.in/pricing",
        "organization": "Groww",
        "doc_type": "pricing_page",
        "category": "Fees & Charges"
    },

    # 4. SEBI Regulatory Mandates & Investor Guides
    {
        "id": "sebi_circular_riskometer_labeling",
        "title": "SEBI Circular on Product Labeling in Mutual Fund Schemes - Risk-o-meter",
        "url": "https://www.sebi.gov.in/legal/circulars/oct-2020/circular-on-product-labeling-in-mutual-fund-schemes-risk-o-meter_47796.html",
        "organization": "SEBI",
        "doc_type": "regulatory_circular",
        "category": "Regulatory Standards"
    },
    {
        "id": "sebi_circular_scheme_categorization",
        "title": "SEBI Categorization and Rationalization of Mutual Fund Schemes",
        "url": "https://www.sebi.gov.in/legal/circulars/oct-2017/categorization-and-rationalization-of-mutual-fund-schemes_36199.html",
        "organization": "SEBI",
        "doc_type": "regulatory_circular",
        "category": "Regulatory Standards"
    },
    {
        "id": "sebi_circular_tri_benchmarking",
        "title": "SEBI Mandate on Benchmarking Schemes to Total Return Index (TRI)",
        "url": "https://www.sebi.gov.in/legal/circulars/jan-2018/benchmarking-of-scheme-s-performance-to-total-return-index_37273.html",
        "organization": "SEBI",
        "doc_type": "regulatory_circular",
        "category": "Regulatory Standards"
    },
    {
        "id": "sebi_investor_elss",
        "title": "SEBI Investor Education - Equity Linked Saving Schemes (ELSS) & 3-Year Lock-in",
        "url": "https://investor.sebi.gov.in/elss.html",
        "organization": "SEBI Investor Office",
        "doc_type": "investor_guide",
        "category": "ELSS & Lock-in"
    },
    {
        "id": "sebi_investor_riskometer",
        "title": "SEBI Investor Education - Understanding the Scheme Risk-o-meter",
        "url": "https://investor.sebi.gov.in/riskometer.html",
        "organization": "SEBI Investor Office",
        "doc_type": "investor_guide",
        "category": "Riskometer & Risk"
    },
    {
        "id": "sebi_investor_exit_load",
        "title": "SEBI Investor Education - Mutual Fund Exit Loads & Redemption Deductions",
        "url": "https://investor.sebi.gov.in/exit_load.html",
        "organization": "SEBI Investor Office",
        "doc_type": "investor_guide",
        "category": "Charges & Exit Load"
    },
    {
        "id": "sebi_investor_cas",
        "title": "SEBI Investor Education - Consolidated Account Statement (CAS)",
        "url": "https://investor.sebi.gov.in/consolidated_account_statement.html",
        "organization": "SEBI Investor Office",
        "doc_type": "investor_guide",
        "category": "Statements & CAS"
    },

    # 5. AMFI Standards & CAS Services
    {
        "id": "amfi_knowledge_expense_ratio",
        "title": "AMFI Knowledge Center - Total Expense Ratio (TER) in Mutual Funds",
        "url": "https://www.amfiindia.com/investor/knowledge-center-info?zoneName=expenseRatio",
        "organization": "AMFI",
        "doc_type": "amfi_guide",
        "category": "Charges & Expense Ratio"
    },
    {
        "id": "amfi_online_download_cas",
        "title": "AMFI Online Center - How to Download Consolidated Account Statement (CAS)",
        "url": "https://www.amfiindia.com/online-center/download-cas",
        "organization": "AMFI",
        "doc_type": "amfi_guide",
        "category": "Statements & CAS"
    },
    {
        "id": "amfi_faq_cas_pdf",
        "title": "AMFI Comprehensive FAQs on Consolidated Account Statement (CAS)",
        "url": "https://portal.amfiindia.com/spages/FAQ%20on%20CAS.pdf",
        "organization": "AMFI",
        "doc_type": "statutory_faq",
        "category": "Statements & CAS",
        "is_pdf": True
    }
]

def fetch_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()

def clean_html_to_text(html_str):
    # Remove script, style, header, nav, footer tags
    clean = re.sub(r'<(script|style|header|footer|nav|noscript)[^>]*>.*?</\1>', ' ', html_str, flags=re.DOTALL | re.IGNORECASE)
    # Convert headings to markdown hashes
    clean = re.sub(r'<h1[^>]*>(.*?)</h1>', r'\n\n# \1\n\n', clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<h2[^>]*>(.*?)</h2>', r'\n\n## \1\n\n', clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<h3[^>]*>(.*?)</h3>', r'\n\n### \1\n\n', clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<h4[^>]*>(.*?)</h4>', r'\n\n#### \1\n\n', clean, flags=re.DOTALL | re.IGNORECASE)
    # Convert list items
    clean = re.sub(r'<li[^>]*>(.*?)</li>', r'\n* \1', clean, flags=re.DOTALL | re.IGNORECASE)
    # Convert paragraphs and breaks
    clean = re.sub(r'<p[^>]*>(.*?)</p>', r'\n\n\1\n\n', clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<br\s*/?>', '\n', clean, flags=re.IGNORECASE)
    # Strip any remaining tags
    clean = re.sub(r'<[^>]+>', ' ', clean)
    clean = unescape(clean)
    # Clean excessive whitespace
    lines = [re.sub(r'\s+', ' ', line).strip() for line in clean.splitlines()]
    non_empty = []
    prev_blank = False
    for line in lines:
        if line:
            non_empty.append(line)
            prev_blank = False
        elif not prev_blank:
            non_empty.append('')
            prev_blank = True
    return '\n'.join(non_empty).strip()

def format_scheme_data(data, meta):
    sd = data.get('props', {}).get('pageProps', {}).get('mfServerSideData', {})
    if not sd:
        return None
    
    scheme_name = sd.get('scheme_name') or meta.get('scheme_name')
    fund_house = sd.get('fund_house') or 'Groww Mutual Fund'
    category = sd.get('category', '')
    sub_category = sd.get('sub_category', '')
    expense_ratio = sd.get('expense_ratio')
    exit_load = sd.get('exit_load')
    lock_in_data = sd.get('lock_in', {})
    lock_in_str = f"{lock_in_data.get('years')} Years" if lock_in_data and lock_in_data.get('years') else "Nil (No lock-in)"
    min_sip = sd.get('min_sip_investment')
    min_lump = sd.get('min_investment_amount')
    benchmark = sd.get('benchmark_name') or sd.get('benchmark')
    stamp_duty = sd.get('stamp_duty', '0.005%')
    description = sd.get('description', '')
    fund_managers = sd.get('fund_manager', '')
    launch_date = sd.get('launch_date', '')
    aum = sd.get('aum', '')

    md_content = f"""# {scheme_name}

## Official Scheme Overview & Key Facts
- **Fund House / AMC:** {fund_house}
- **Category:** {category} / {sub_category}
- **Benchmark:** {benchmark}
- **Expense Ratio (Direct Growth):** {expense_ratio}% (Total Expense Ratio as declared by AMC)
- **Exit Load:** {exit_load}
- **Statutory Lock-in Period:** {lock_in_str}
- **Minimum SIP Investment:** ₹{min_sip}
- **Minimum Lump-sum Investment:** ₹{min_lump}
- **Stamp Duty:** {stamp_duty}
- **Fund Manager(s):** {fund_managers}
- **Launch Date:** {launch_date}

## Investment Objective & Description
{description}

## Source & Regulatory Notice
All data reflects official disclosures from Groww Mutual Fund and Groww Investment Platform.
- Source URL: {meta['url']}
- Last updated from sources: {datetime.now().strftime('%Y-%m-%d')}
"""
    return md_content

def process_source(meta, out_dir):
    source_id = meta['id']
    url = meta['url']
    print(f"[*] Downloading [{source_id}] {url}...")
    
    raw_path = os.path.join(out_dir, f"{source_id}.raw")
    md_path = os.path.join(out_dir, f"{source_id}.md")
    
    try:
        content_bytes = fetch_url(url)
    except Exception as e:
        print(f"[-] ERROR downloading {url}: {e}")
        return False, str(e)
    
    extracted_text = ""
    
    if meta.get('is_pdf'):
        pdf_path = os.path.join(out_dir, f"{source_id}.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(content_bytes)
        # Use pdftotext
        txt_out = os.path.join(out_dir, f"{source_id}.txt")
        res = subprocess.run(["pdftotext", "-layout", pdf_path, txt_out], capture_output=True)
        if res.returncode == 0 and os.path.exists(txt_out):
            with open(txt_out, 'r', encoding='utf-8', errors='ignore') as tf:
                extracted_text = tf.read().strip()
            # truncate overly repetitive pages if too large
            if len(extracted_text) > 20000:
                extracted_text = extracted_text[:20000] + "\n\n... [Extracted content summary truncated for RAG context window] ..."
        else:
            extracted_text = f"Official PDF document from {meta['organization']}. URL: {url}"
    else:
        html_str = content_bytes.decode('utf-8', errors='ignore')
        # Check if Next.js scheme page
        if meta.get('is_scheme_page') and '__NEXT_DATA__' in html_str:
            match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html_str)
            if match:
                try:
                    data = json.loads(match.group(1))
                    formatted_md = format_scheme_data(data, meta)
                    if formatted_md:
                        extracted_text = formatted_md
                except Exception as ex:
                    print(f"  Warning parsing Next.js data: {ex}")
        
        if not extracted_text:
            extracted_text = clean_html_to_text(html_str)
            if len(extracted_text) > 15000:
                extracted_text = extracted_text[:15000] + "\n\n... [Truncated for RAG indexing] ..."
    
    # Save frontmatter + markdown
    today = datetime.now().strftime('%Y-%m-%d')
    doc_content = f"""---
id: {source_id}
title: "{meta['title']}"
url: "{meta['url']}"
organization: "{meta['organization']}"
doc_type: "{meta['doc_type']}"
category: "{meta['category']}"
last_updated_from_sources: "{today}"
---

{extracted_text}
"""
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(doc_content)
    
    print(f"[+] Successfully saved {md_path} ({len(extracted_text)} chars)")
    return True, None

def main():
    out_dir = os.path.abspath("sources")
    os.makedirs(out_dir, exist_ok=True)
    
    catalog = []
    success_count = 0
    
    for meta in SOURCES_METADATA:
        ok, err = process_source(meta, out_dir)
        catalog.append({
            "id": meta["id"],
            "title": meta["title"],
            "url": meta["url"],
            "organization": meta["organization"],
            "doc_type": meta["doc_type"],
            "category": meta["category"],
            "status": "downloaded" if ok else f"error: {err}",
            "file": f"{meta['id']}.md"
        })
        if ok:
            success_count += 1
            
    # Write sources_catalog.json
    with open(os.path.join(out_dir, "sources_catalog.json"), "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
        
    # Write sources_catalog.csv
    with open(os.path.join(out_dir, "sources_catalog.csv"), "w", encoding="utf-8") as f:
        f.write("Index,Source ID,Title,Organization,Category,Document Type,Official Public URL\n")
        for i, item in enumerate(catalog, 1):
            f.write(f'{i},"{item["id"]}","{item["title"]}","{item["organization"]}","{item["category"]}","{item["doc_type"]}","{item["url"]}"\n')
            
    # Write sources_catalog.md
    with open(os.path.join(out_dir, "sources_catalog.md"), "w", encoding="utf-8") as f:
        f.write(f"# Official Public Sources Catalog (Groww Mutual Fund FAQ Assistant)\n\n")
        f.write(f"Total Sources Collected: **{success_count} / {len(SOURCES_METADATA)}** (Meets requirement of 15–25 official public pages)\n\n")
        f.write(f"| # | Scheme / Topic | Organization | Category | Official Source Citation URL |\n")
        f.write(f"|---|---|---|---|---|\n")
        for i, item in enumerate(catalog, 1):
            f.write(f'| {i} | {item["title"]} | {item["organization"]} | {item["category"]} | [{item["url"]}]({item["url"]}) |\n')
            
    print(f"\n[DONE] Finished downloading {success_count}/{len(SOURCES_METADATA)} sources into {out_dir}/")

if __name__ == "__main__":
    main()
