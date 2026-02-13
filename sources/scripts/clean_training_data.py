#!/usr/bin/env python3
"""
Clean training data file by removing extraneous content.

Removes:
- Reddit metadata headers (# Reddit Post, # Score:, # URL:, etc.)
- Standalone headlines (not in QUESTION: format)
- Columnist names and references (Carolyn, Hax, Ned, Nick, etc.)
- Reddit bot boilerplate
- Reddit update/TLDR sections
- URLs and links
- Number-only lines
- Column meta-commentary
- "Re:" prefix lines
- Separator lines (===, ---)

Usage:
    python3 clean_training_data.py --input sources/training_data_final_merged.md [--remove]
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import shutil


class TrainingDataCleaner:
    def __init__(self, input_file: Path, remove: bool = False):
        self.input_file = Path(input_file)
        self.remove = remove
        self.issues_found = {
            "reddit_metadata": [],
            "standalone_headlines": [],
            "columnist_references": [],
            "reddit_bot_boilerplate": [],
            "reddit_updates": [],
            "urls": [],
            "number_only_lines": [],
            "column_meta": [],
            "re_prefix_lines": [],
            "separators": [],
            "meaningless_answers": [],
            "technical_questions": [],
            "headline_continuations": [],
        }
        
    def is_reddit_metadata(self, line: str) -> bool:
        """Check if line is Reddit metadata header"""
        patterns = [
            r"^# Reddit Post \d+",
            r"^# Score:",
            r"^# URL:",
            r"^# Date:",
        ]
        return any(re.match(pattern, line) for pattern in patterns)
    
    def is_separator(self, line: str) -> bool:
        """Check if line is a separator (=== or ---)"""
        stripped = line.strip()
        return stripped and all(c in "=-" for c in stripped) and len(stripped) >= 3
    
    def is_standalone_headline(self, line: str, prev_line: str, next_line: str) -> bool:
        """Check if line is a standalone headline (not part of QUESTION:)"""
        stripped = line.strip()
        if not stripped:
            return False
        
        # Skip if it's already a QUESTION: or ANSWER: line
        if stripped.startswith("QUESTION:") or stripped.startswith("ANSWER:"):
            return False
        
        # Skip if previous line is QUESTION: (it's part of the question)
        if prev_line and prev_line.strip().startswith("QUESTION:"):
            return False
        
        # Don't skip if previous line is ANSWER: - standalone headlines can appear after answers
        # (they're Reddit post titles that appear between Q&A pairs)
        
        # Patterns that indicate standalone headlines
        headline_patterns = [
            r"^[A-Z][a-z]+'s [A-Za-z]+",  # "Husband's therapist" or "Husband's Therapist"
            r"^[A-Z][a-z]+ [a-z]+ [a-z]+",  # "Needing a break"
            r"^[A-Z][a-z]+ [a-z]+ [a-z]+ [a-z]+",  # "How do I move on"
            r"^[A-Z][a-z]+ [a-z]+ [a-z]+ [a-z]+ [a-z]+",  # Longer headlines
            r"^[A-Z][a-z]+$",  # Single word capitalized (like "Money", "Ned")
            r"^[A-Z][a-z]+ [a-z]+$",  # Two words (like "Go Ask", "Communication styles")
            # Lowercase headlines (often Reddit post titles)
            r"^[a-z]+ [a-z]+ [a-z]+",  # "where is the line"
            r"^[a-z]+ [a-z]+ [a-z]+ [a-z]+",  # Longer lowercase headlines
            r"^where is the",  # Specific pattern for "where is the line/communication style"
            # Standalone comments/phrases that are clearly not questions
            r"^Think I'm",  # "Think I'm just off to kiss my husband."
        ]
        
        # Check if it matches headline patterns
        if any(re.match(pattern, stripped) for pattern in headline_patterns):
            # If next line starts with QUESTION: or ANSWER:, it's likely a standalone headline
            if next_line and (next_line.strip().startswith("QUESTION:") or 
                            next_line.strip().startswith("ANSWER:")):
                return True
            # If it's a short line (likely a title) and previous was blank/separator/ANSWER
            if len(stripped) < 100:
                prev_stripped = prev_line.strip() if prev_line else ""
                if (not prev_stripped or 
                    self.is_separator(prev_line) or 
                    prev_stripped.startswith("ANSWER:")):
                    return True
        
        return False
    
    def is_columnist_reference(self, line: str) -> bool:
        """Check if line contains columnist references"""
        stripped = line.strip()
        
        # Check for lines starting with columnist names
        start_patterns = [
            r"^Carolyn",
            r"^Hax",
            r"^Carolyn Hax",
            r"^--Mystified",
            r"^Ned['s]?",
            r"^Neddikins",
            r"^Nedster",
            r"^NED NED",
            r"^Nick['s]?",
            r"Carolyn Hax chat:",
        ]
        if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in start_patterns):
            return True
        
        # Also check if QUESTION: lines are about Ned (the dog) or other columnist references
        if stripped.startswith("QUESTION:"):
            question_text = stripped[9:].strip()  # Remove "QUESTION:" prefix
            # Check for Ned references in questions (Ned is a dog, not relevant for training)
            ned_patterns = [
                r"\bNed\b",
                r"Ned vid",
                r"Ned video",
                r"Ned's",
                r"see Ned",
                r"show Ned",
            ]
            if any(re.search(pattern, question_text, re.IGNORECASE) for pattern in ned_patterns):
                return True
        
        return False
    
    def is_reddit_bot_boilerplate(self, lines: List[str], idx: int) -> bool:
        """Check if current line is part of Reddit bot boilerplate section"""
        if idx >= len(lines):
            return False
        
        current_line = lines[idx]
        line_lower = current_line.lower()
        
        # Look for bot boilerplate markers in the CURRENT line
        bot_markers = [
            "welcome to /r/",
            "we do not allow",
            "this is an automatic comment",
            "i am a bot",
            "please contact the moderators",
            "please [message the mods]",
            "what we cannot give advice on:",
            "all bans in this subreddit",
            "anyone found to be directly messaging",
            "any sort of namecalling",
            "no referencing hateful subreddits",
            "all advice given must be good, ethical advice",
            "joke advice or tips that serve no purpose",
            "direct-spam account",
        ]
        
        # Check if current line contains bot markers
        if any(marker in line_lower for marker in bot_markers):
            return True
        
        # Also check if we're in a known bot section (look backwards for context)
        # Bot sections typically start with "ANSWER: Welcome to /r/" or similar
        # and end with "I am a bot" or separator
        if idx > 0:
            # Look back up to 5 lines to see if we're in a bot section
            for i in range(max(0, idx - 5), idx):
                prev_line_lower = lines[i].lower()
                if "welcome to /r/" in prev_line_lower or "i am a bot" in prev_line_lower:
                    # Check if current line is likely part of bot section
                    # (empty lines, separators, or continuation of bot text)
                    if (not current_line.strip() or 
                        self.is_separator(current_line) or
                        "do not allow" in line_lower or
                        "automatic comment" in line_lower):
                        return True
        
        return False
    
    def is_reddit_update_section(self, line: str) -> bool:
        """Check if line is part of Reddit update/TLDR section"""
        patterns = [
            r"^TLDR:",
            r"^TL;DR:",
            r"^UPDATE:",
            r"^EDIT:",
            r"^Original:",
            r"^Link to original",
            r"^Update:",
            r"^Edit:",
        ]
        stripped = line.strip()
        return any(re.match(pattern, stripped, re.IGNORECASE) for pattern in patterns)
    
    def contains_url(self, line: str) -> bool:
        """Check if line contains URLs or markdown links"""
        url_patterns = [
            r"https?://[^\s]+",
            r"\[.*?\]\(https?://[^\)]+\)",
            r"reddit\.com/r/",
        ]
        return any(re.search(pattern, line) for pattern in url_patterns)

    def is_junk_url_line(self, line: str) -> bool:
        """
        Check if a URL-containing line is mostly tracking/junk rather than useful content.

        We keep most URLs for context, but strip obvious tracking domains and long
        tracking query strings that don't help the model learn advice patterns.
        """
        line_lower = line.lower()
        junk_markers = [
            "redditgiftsquiz.com",
            "utm_source=",
            "utm_medium=",
            "utm_campaign=",
            "utm_term=",
            "utm_content=",
        ]
        if any(marker in line_lower for marker in junk_markers):
            return True

        return False
    
    def is_number_only(self, line: str) -> bool:
        """Check if line is just a number"""
        stripped = line.strip()
        if not stripped:
            return False
        # Check if it's just digits (possibly with commas)
        return bool(re.match(r"^\d+$", stripped.replace(",", "")))
    
    def is_column_meta(self, line: str) -> bool:
        """Check if line is column meta-commentary"""
        meta_patterns = [
            r"Sun, Tues, Thurs new",
            r"Mon, Sat, adapted",
            r"Wed, readers",
            r"Fri, chat",
            r"I will be off",
            r"Sorry I was a couple of minutes late",
            r"Wait, there's more to my intro",
            r"Your column",
            r"the chat",
            r"the column",
            r"production commitment",
            r"cartoons and daily columns",
            r"too big a file",
        ]
        line_lower = line.lower()
        return any(pattern.lower() in line_lower for pattern in meta_patterns)
    
    def is_re_prefix(self, line: str) -> bool:
        """Check if line starts with Re: prefix (standalone)"""
        stripped = line.strip()
        if not stripped.startswith("Re:") and not stripped.startswith("re:"):
            return False
        # If it's a standalone line (not part of QUESTION:), flag it
        # But allow "Re: [topic]" if it's part of a question
        if len(stripped) < 50 and not stripped.startswith("QUESTION:"):
            return True
        return False
    
    def is_meaningless_answer(self, line: str) -> bool:
        """Check if answer is meaningless (single word, very short, etc.)"""
        stripped = line.strip()
        if not stripped.startswith("ANSWER:"):
            return False
        
        # Extract the answer content (after "ANSWER:")
        answer_content = stripped[7:].strip()  # Remove "ANSWER:" prefix
        
        # Check for meaningless single-word answers
        meaningless_words = ["and", "or", "but", "the", "a", "an", "it", "is", "was", "are", "were"]
        if answer_content.lower() in meaningless_words:
            return True
        
        # Check for very short answers (less than 10 characters, excluding punctuation)
        answer_clean = re.sub(r'[^\w\s]', '', answer_content)
        if len(answer_clean.strip()) < 10 and len(answer_content) < 20:
            return True
        
        return False
    
    def is_technical_question(self, line: str) -> bool:
        """Check if question is about technical issues (browser, video, etc.)"""
        stripped = line.strip()
        if not stripped.startswith("QUESTION:"):
            return False
        
        question_text = stripped[9:].strip().lower()  # Remove "QUESTION:" prefix
        
        # More specific technical issue patterns (avoid false positives)
        # These patterns indicate questions about website/browser/video technical issues
        technical_patterns = [
            r"\bbrowser\b.*working",  # "browser" and "working" together
            r"not working.*browser",  # "not working" with "browser"
            r"chrome|firefox|safari|edge.*not working",  # Specific browser + not working
            r"video.*not playing",  # Video playback issues
            r"not playing.*video",  # Video playback issues
            r"refresh.*button",  # Refresh button issues
            r"button.*works",  # Button functionality
            r"click.*not working",  # Click functionality
            r"website.*broken",  # Website issues
            r"page.*loading",  # Page loading issues
            r"link.*broken",  # Broken links
            r"error.*page",  # Error pages
            r"what browser",  # Direct browser questions
            r"which browser",  # Browser selection questions
        ]
        
        # Check for specific technical question patterns
        if any(re.search(pattern, question_text) for pattern in technical_patterns):
            return True
        
        # Also check for very short technical questions (likely about UI/technical issues)
        if len(question_text) < 100:
            short_tech_keywords = ["browser", "chrome", "firefox", "safari", "edge", "video", "button", "click"]
            if any(keyword in question_text for keyword in short_tech_keywords):
                return True
        
        return False
    
    def is_headline_continuation(self, line: str, prev_line: str, lines: List[str], idx: int) -> bool:
        """Check if line is a continuation of a headline (like "click for more")"""
        stripped = line.strip()
        stripped_lower = stripped.lower()
        
        # Skip if it's a QUESTION: or ANSWER: line
        if stripped.startswith("QUESTION:") or stripped.startswith("ANSWER:"):
            return False
        
        # Look back past blank lines to find the previous QUESTION: or ANSWER:
        prev_non_blank = None
        for i in range(idx - 1, max(-1, idx - 10), -1):  # Look back up to 10 lines
            if i >= 0 and lines[i].strip():
                prev_non_blank = lines[i].strip()
                break
        
        # Skip if we found a QUESTION: or ANSWER: in recent lines (it's part of that Q&A, not a headline)
        if prev_non_blank and (prev_non_blank.startswith("QUESTION:") or prev_non_blank.startswith("ANSWER:")):
            return False
        
        # Skip if line looks like it's part of a question/answer (starts with "I (", "My ", etc.)
        # These are common question/answer continuations, not headlines
        if re.match(r'^[A-Z][a-z]+ \([0-9]+[MF]\)', stripped):
            return False
        if re.match(r'^[A-Z][a-z]+ (was|is|am|are|have|had)', stripped):
            return False
        # Skip long lines that are clearly question/answer content, not headlines
        if len(stripped) > 100:
            return False
        
        # Common headline continuation phrases (exact matches or contained)
        continuation_phrases = [
            "click for more",
            "read more",
            "see more",
            "continue reading",
            "more below",
            '"click for more"',
            '(click for more)',
        ]
        
        if any(phrase in stripped_lower for phrase in continuation_phrases):
            return True
        
        # Check for quoted single-character or very short phrases (likely UI elements)
        # But only if they're very short (1-3 characters)
        if stripped.startswith('"') and stripped.endswith('"') and len(stripped) <= 5:
            # Only catch very short ones like "K"
            if len(stripped) <= 5:
                return True
        
        # Check for parenthesized short phrases that are clearly UI elements
        if stripped.startswith('(') and stripped.endswith(')') and len(stripped) < 30:
            if any(phrase in stripped_lower for phrase in ["click", "more", "read"]):
                return True
        
        return False
    
    def analyze_file(self) -> Tuple[List[str], dict]:
        """Analyze file and return cleaned lines and issues found"""
        with open(self.input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        cleaned_lines = []
        in_bot_section = False
        
        # Track Q&A pairs: if we remove a column_meta QUESTION, also remove its ANSWER
        in_question_to_remove = False
        
        for idx, line in enumerate(lines):
            prev_line = lines[idx - 1] if idx > 0 else ""
            next_line = lines[idx + 1] if idx < len(lines) - 1 else ""
            
            should_remove = False
            issue_type = None
            
            # Check for Reddit metadata
            if self.is_reddit_metadata(line):
                should_remove = True
                issue_type = "reddit_metadata"
            
            # Check for separators
            elif self.is_separator(line):
                should_remove = True
                issue_type = "separators"
            
            # Check for standalone headlines
            elif self.is_standalone_headline(line, prev_line, next_line):
                should_remove = True
                issue_type = "standalone_headlines"
            
            # Check for columnist references
            elif self.is_columnist_reference(line):
                should_remove = True
                issue_type = "columnist_references"
            
            # Check for Reddit bot boilerplate
            elif self.is_reddit_bot_boilerplate(lines, idx):
                if not in_bot_section:
                    in_bot_section = True
                should_remove = True
                issue_type = "reddit_bot_boilerplate"
                # Check if we should end bot section (blank line after bot content)
                if idx < len(lines) - 1 and not lines[idx + 1].strip():
                    in_bot_section = False
            
            # Check for Reddit update sections
            elif self.is_reddit_update_section(line):
                should_remove = True
                issue_type = "reddit_updates"
            
            # Selectively remove junk/tracking URLs while keeping useful links
            elif self.contains_url(line) and self.is_junk_url_line(line):
                should_remove = True
                issue_type = "urls"
            
            # Check for number-only lines
            elif self.is_number_only(line):
                should_remove = True
                issue_type = "number_only_lines"
            
            # Check for column meta-commentary
            elif self.is_column_meta(line):
                # If this is a QUESTION: line about column meta, mark to remove entire Q&A pair
                if line.strip().startswith("QUESTION:"):
                    should_remove = True
                    issue_type = "column_meta"
                    in_question_to_remove = True  # Flag to remove the entire Q&A pair
                else:
                    should_remove = True
                    issue_type = "column_meta"
            
            # If we're removing a technical question, also remove its answer
            # (Technical questions are not useful for training)
            if should_remove and issue_type == "technical_questions":
                in_question_to_remove = True
            
            # If we're in the middle of removing a Q&A pair (column_meta or technical), continue removing
            elif in_question_to_remove:
                stripped = line.strip()
                # Stop removing when we hit the next QUESTION:
                if stripped.startswith("QUESTION:"):
                    # New question starts, stop removing this pair
                    in_question_to_remove = False
                    # Check if this new question should also be removed (recursive check)
                    if self.is_column_meta(line):
                        should_remove = True
                        issue_type = "column_meta"
                        in_question_to_remove = True
                    elif self.is_technical_question(line):
                        should_remove = True
                        issue_type = "technical_questions"
                        in_question_to_remove = True
                    else:
                        should_remove = False
                else:
                    # Continue removing lines in this Q&A pair (answer and any continuation)
                    # Use the original issue type (column_meta or technical_questions)
                    should_remove = True
                    if issue_type is None:
                        issue_type = "technical_questions"  # Default if we lost track
            
            # Check for Re: prefix lines
            elif self.is_re_prefix(line):
                should_remove = True
                issue_type = "re_prefix_lines"
            
            # Check for meaningless answers (single word, very short)
            elif self.is_meaningless_answer(line):
                should_remove = True
                issue_type = "meaningless_answers"
            
            # Check for technical questions (browser, video, etc.)
            elif self.is_technical_question(line):
                should_remove = True
                issue_type = "technical_questions"
            
            # Check for headline continuations
            elif self.is_headline_continuation(line, prev_line, lines, idx):
                should_remove = True
                issue_type = "headline_continuations"
            
            # Track issues
            if should_remove and issue_type:
                self.issues_found[issue_type].append((idx + 1, line.rstrip()))
            
            # Add line to cleaned_lines if it should be kept
            # (We always build cleaned_lines to estimate size, even in identify mode)
            if not should_remove:
                cleaned_lines.append(line)
            # If should_remove is True, we skip adding it (whether in remove mode or not)
        
        return cleaned_lines, self.issues_found
    
    def generate_report(self) -> str:
        """Generate a report of issues found"""
        report = []
        report.append("=" * 80)
        report.append("TRAINING DATA CLEANUP REPORT")
        report.append("=" * 80)
        report.append(f"File: {self.input_file}")
        report.append(f"Mode: {'REMOVE' if self.remove else 'IDENTIFY ONLY'}")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        total_issues = sum(len(issues) for issues in self.issues_found.values())
        report.append(f"Total issues found: {total_issues}")
        report.append("")
        
        for issue_type, issues in self.issues_found.items():
            if issues:
                report.append(f"\n{issue_type.replace('_', ' ').title()}: {len(issues)} issues")
                # Show first 5 examples
                for line_num, line_content in issues[:5]:
                    preview = line_content[:70] + "..." if len(line_content) > 70 else line_content
                    report.append(f"  Line {line_num}: {preview}")
                if len(issues) > 5:
                    report.append(f"  ... and {len(issues) - 5} more")
        
        return "\n".join(report)
    
    def clean(self) -> None:
        """Clean the file"""
        print("Analyzing file...")
        cleaned_lines, issues = self.analyze_file()
        
        # Calculate estimated size reduction
        original_size = self.input_file.stat().st_size
        # Estimate new size by calculating bytes in cleaned lines
        estimated_new_size = sum(len(line.encode('utf-8')) for line in cleaned_lines)
        estimated_reduction = ((original_size - estimated_new_size) / original_size) * 100
        
        # Generate and print report
        report = self.generate_report()
        print("\n" + report)
        
        # Print size estimates
        print("\n📊 Size Estimates:")
        print(f"   Current size: {original_size / 1024:.1f} KB ({original_size / (1024*1024):.2f} MB)")
        print(f"   Estimated new size: {estimated_new_size / 1024:.1f} KB ({estimated_new_size / (1024*1024):.2f} MB)")
        print(f"   Estimated reduction: {estimated_reduction:.1f}% ({(original_size - estimated_new_size) / 1024:.1f} KB)")
        
        if not self.remove:
            print("\n⚠️  Running in IDENTIFY mode. Use --remove to actually clean the file.")
            return
        
        # Create backup
        backup_file = Path(f"backups/training_data_final_merged_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        print(f"\n📦 Creating backup: {backup_file}")
        shutil.copy2(self.input_file, backup_file)
        
        # Write cleaned file
        print(f"✏️  Writing cleaned file: {self.input_file}")
        with open(self.input_file, "w", encoding="utf-8") as f:
            f.writelines(cleaned_lines)
        
        # Calculate actual stats
        new_size = self.input_file.stat().st_size
        actual_reduction = ((original_size - new_size) / original_size) * 100
        
        print("\n✅ Cleanup complete!")
        print(f"   Original size: {original_size / 1024:.1f} KB ({original_size / (1024*1024):.2f} MB)")
        print(f"   New size: {new_size / 1024:.1f} KB ({new_size / (1024*1024):.2f} MB)")
        print(f"   Actual reduction: {actual_reduction:.1f}% ({(original_size - new_size) / 1024:.1f} KB)")
        print(f"   Backup saved: {backup_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean training data by removing extraneous content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input training data file",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Actually remove the issues (default: identify only)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Save report to file (optional)",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Error: File not found: {args.input}")
        return 1
    
    cleaner = TrainingDataCleaner(args.input, remove=args.remove)
    cleaner.clean()
    
    # Save report if requested
    if args.output_report:
        report = cleaner.generate_report()
        with open(args.output_report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📄 Report saved to: {args.output_report}")
    
    return 0


if __name__ == "__main__":
    exit(main())

