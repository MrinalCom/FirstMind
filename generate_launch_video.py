#!/usr/bin/env python3
"""
FirstMind Cinematic Launch Video Generator (With Real Voiceover & Beat)
Renders a 32-second, 30fps Full HD (1920x1080) modern tech product launch motion video.
Features:
- Real professional voiceover narration (Daniel - UK English)
- High-energy electronic music bed (808 kick, snappy snare, 16th hats, synth bass & arpeggio)
- Master audio mixing with voice compression & music ducking
- macOS Glassmorphic UI simulation with live typing and streaming tokens
- Perplexity / ChatGPT Search comparison: Before (0% presence) vs After (Rank #1)
- DeepSeek GRPO multi-rollout architecture visualization
- Kinetic telemetry metrics (86.7% APR, 80% Win Rate, 61.9% SoM)
- Cyberpunk grid, floating neon particles, and glowing badges
"""

import os
import sys
import math
import subprocess
import random
import wave
import struct
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 1080
FPS = 30
TOTAL_SECONDS = 32
TOTAL_FRAMES = FPS * TOTAL_SECONDS
SAMPLE_RATE = 44100
BPM = 124
BEAT_SEC = 60.0 / BPM

OUTPUT_MP4 = "/Volumes/Mrinal/RL/FirstMind_Official_Launch_Video.mp4"

# Load Fonts
def get_font(size, bold=False, mono=False):
    try:
        if mono:
            return ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size)
        if bold:
            return ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", size, index=1)
        return ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", size, index=0)
    except Exception:
        try:
            if mono:
                return ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", size)
            if bold:
                return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
            return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

# Pre-load fonts with tailored sizes
FONT_HERO = get_font(72, bold=True)
FONT_HERO_MEDIUM = get_font(48, bold=True)
FONT_TITLE = get_font(38, bold=True)
FONT_TITLE_SMALL = get_font(28, bold=True)
FONT_SUBTITLE = get_font(26, bold=False)
FONT_BODY = get_font(21, bold=False)
FONT_BODY_BOLD = get_font(21, bold=True)
FONT_CODE = get_font(19, mono=True)
FONT_CODE_BOLD = get_font(19, mono=True)
FONT_STAT_NUM = get_font(80, bold=True)
FONT_STAT_LBL = get_font(19, bold=True)
FONT_TAG = get_font(15, bold=True)

# Generate pseudo-random particles
random.seed(42)
NUM_PARTICLES = 45
particles = []
palette = [(99, 102, 241), (6, 182, 212), (16, 185, 129), (236, 72, 153)]
for _ in range(NUM_PARTICLES):
    particles.append({
        'x': random.uniform(0, WIDTH),
        'y': random.uniform(0, HEIGHT),
        'vx': random.uniform(-0.5, 0.5),
        'vy': random.uniform(-0.8, -0.2),
        'size': random.uniform(2, 5),
        'alpha': random.uniform(40, 160),
        'color': random.choice(palette)
    })

def draw_background(draw, frame_idx, glow_center=(960, 540), glow_color=(45, 55, 120)):
    # Base dark gradient
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(8, 10, 16, 255))
    
    # Perspective Grid
    grid_spacing = 60
    grid_y_offset = int((frame_idx * 1.5) % grid_spacing)
    for x in range(0, WIDTH, grid_spacing):
        draw.line([(x, 0), (x, HEIGHT)], fill=(20, 26, 40, 60), width=1)
    for y in range(-grid_spacing + grid_y_offset, HEIGHT, grid_spacing):
        draw.line([(0, y), (WIDTH, y)], fill=(20, 26, 40, 60), width=1)
        
    # Ambient radial light ring
    gx, gy = glow_center
    pulse = math.sin(frame_idx * 0.08) * 35
    for r in range(400, 50, -50):
        alpha = int(max(0, 12 - (r / 40)))
        draw.ellipse([gx - r - pulse, gy - r - pulse, gx + r + pulse, gy + r + pulse],
                     outline=(glow_color[0], glow_color[1], glow_color[2], alpha), width=3)

    # Floating particles
    for p in particles:
        px = (p['x'] + p['vx'] * frame_idx) % WIDTH
        py = (p['y'] + p['vy'] * frame_idx) % HEIGHT
        alpha = int(p['alpha'] * (0.6 + 0.4 * math.sin(frame_idx * 0.05 + px)))
        c = p['color']
        draw.ellipse([px, py, px + p['size'], py + p['size']], fill=(c[0], c[1], c[2], alpha))

def draw_mac_window(draw, x1, y1, x2, y2, title="perplexity.ai/search"):
    # Outer glow / shadow border
    draw.rounded_rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], radius=16,
                           outline=(50, 65, 105, 180), width=2)
    # Window body
    draw.rounded_rectangle([x1, y1, x2, y2], radius=16, fill=(15, 19, 30, 245))
    
    # Window header bar
    draw.rounded_rectangle([x1, y1, x2, y1 + 52], radius=16, fill=(22, 28, 44, 255))
    draw.rectangle([x1, y1 + 32, x2, y1 + 52], fill=(22, 28, 44, 255))
    draw.line([(x1, y1 + 52), (x2, y1 + 52)], fill=(40, 50, 75, 255), width=1)
    
    # Traffic light buttons
    draw.ellipse([x1 + 22, y1 + 18, x1 + 36, y1 + 32], fill=(255, 95, 86, 255)) # Red
    draw.ellipse([x1 + 44, y1 + 18, x1 + 58, y1 + 32], fill=(255, 189, 46, 255)) # Yellow
    draw.ellipse([x1 + 66, y1 + 18, x1 + 80, y1 + 32], fill=(39, 201, 63, 255)) # Green
    
    # URL bar in center
    url_w = 460
    url_x1 = (x1 + x2) // 2 - url_w // 2
    url_x2 = url_x1 + url_w
    draw.rounded_rectangle([url_x1, y1 + 10, url_x2, y1 + 42], radius=8, fill=(10, 13, 22, 255),
                           outline=(35, 45, 70, 255), width=1)
    # Lock indicator + URL text
    draw.rounded_rectangle([url_x1 + 14, y1 + 20, url_x1 + 24, y1 + 32], radius=2, fill=(16, 185, 129, 255))
    draw.text((url_x1 + 34, y1 + 16), "https://" + title, font=FONT_TAG, fill=(140, 155, 185, 255))

# ==========================================
# SCENE 1: THE HOOK (0.0s – 5.0s)
# ==========================================
def render_scene_1(draw, frame_idx):
    sec = frame_idx / FPS
    draw_background(draw, frame_idx, glow_center=(960, 540), glow_color=(220, 50, 60) if sec < 3.4 else (99, 102, 241))
    
    # Top Brand Kicker
    draw.rounded_rectangle([800, 180, 1120, 225], radius=20, fill=(30, 20, 35, 200), outline=(239, 68, 68, 200), width=1)
    draw.text((820, 192), "// CRITICAL SHIFT IN BUYER DISCOVERY", font=FONT_TAG, fill=(248, 113, 113, 255))
    
    if sec < 1.8:
        # Phase 1: Search is dead
        alpha = int(min(255, sec * 400))
        text = "SEARCH ENGINES ARE EXTINCT."
        bbox = draw.textbbox((0, 0), text, font=FONT_HERO)
        tw = bbox[2] - bbox[0]
        draw.text(((WIDTH - tw) // 2, 380), text, font=FONT_HERO, fill=(255, 255, 255, alpha))
        
        sub = "Traditional SEO ranking keywords is officially obsolete."
        sbbox = draw.textbbox((0, 0), sub, font=FONT_SUBTITLE)
        draw.text(((WIDTH - (sbbox[2]-sbbox[0])) // 2, 480), sub, font=FONT_SUBTITLE, fill=(160, 175, 205, alpha))
        
        draw.rounded_rectangle([700, 600, 1220, 680], radius=12, fill=(40, 15, 20, 240), outline=(239, 68, 68, 255), width=2)
        draw.text((730, 626), "78% OF B2B BUYERS NOW ASK AI ENGINES FIRST", font=FONT_BODY_BOLD, fill=(252, 165, 165, 255))
        
    elif sec < 3.4:
        # Phase 2: AI answers only
        text = "BUYERS ASK PERPLEXITY & CHATGPT."
        bbox = draw.textbbox((0, 0), text, font=FONT_HERO)
        tw = bbox[2] - bbox[0]
        draw.text(((WIDTH - tw) // 2, 380), text, font=FONT_HERO, fill=(6, 182, 212, 255))
        
        sub = "Direct answers. Synthesized recommendations. Zero blue links."
        sbbox = draw.textbbox((0, 0), sub, font=FONT_SUBTITLE)
        draw.text(((WIDTH - (sbbox[2]-sbbox[0])) // 2, 480), sub, font=FONT_SUBTITLE, fill=(160, 175, 205, 255))
        
        draw.rounded_rectangle([740, 600, 1180, 680], radius=12, fill=(15, 30, 45, 240), outline=(6, 182, 212, 255), width=2)
        draw.text((775, 626), "ONLY TOP 3 BRANDS GET CITED", font=FONT_BODY_BOLD, fill=(103, 232, 249, 255))
        
    else:
        # Phase 3: The devastating question
        text = "IF AI DOES NOT CITE YOU..."
        bbox = draw.textbbox((0, 0), text, font=FONT_HERO)
        tw = bbox[2] - bbox[0]
        draw.text(((WIDTH - tw) // 2, 360), text, font=FONT_HERO, fill=(255, 255, 255, 255))
        
        sub = "YOU DO NOT EXIST."
        sbbox = draw.textbbox((0, 0), sub, font=FONT_HERO)
        draw.text(((WIDTH - (sbbox[2]-sbbox[0])) // 2, 470), sub, font=FONT_HERO, fill=(239, 68, 68, 255))

# =======================================================
# SCENE 2: THE PROBLEM (PERPLEXITY FAILURE) (5.0s – 11.0s)
# =======================================================
def render_scene_2(draw, frame_idx):
    sec = frame_idx / FPS
    rel_sec = sec - 5.0
    draw_background(draw, frame_idx, glow_center=(960, 520), glow_color=(200, 40, 50))
    
    # Top banner
    draw.rounded_rectangle([660, 50, 1260, 95], radius=16, fill=(35, 15, 20, 220), outline=(239, 68, 68, 200), width=1)
    draw.text((700, 62), "// BEFORE GEO: UNTRAINED BASELINE RESPONSE", font=FONT_TAG, fill=(252, 165, 165, 255))
    
    # Render Mac Window
    win_x1, win_y1, win_x2, win_y2 = 220, 120, 1700, 960
    draw_mac_window(draw, win_x1, win_y1, win_x2, win_y2, title="perplexity.ai/search")
    
    # Search input bar
    query_text = "What are the top SOC 2 compliance tools for fintech startups on AWS?"
    chars_to_show = int(min(len(query_text), rel_sec * 28))
    current_query = query_text[:chars_to_show]
    
    draw.rounded_rectangle([win_x1 + 60, win_y1 + 80, win_x2 - 60, win_y1 + 145], radius=12,
                           fill=(22, 28, 44, 255), outline=(60, 75, 110, 255), width=2)
    # Search icon vector
    draw.ellipse([win_x1 + 85, win_y1 + 102, win_x1 + 101, win_y1 + 118], outline=(140, 155, 185, 255), width=2)
    draw.line([(win_x1 + 97, win_y1 + 115), (win_x1 + 105, win_y1 + 123)], fill=(140, 155, 185, 255), width=2)
    
    draw.text((win_x1 + 120, win_y1 + 98), current_query, font=FONT_BODY_BOLD, fill=(240, 245, 255, 255))
    
    # Blinking cursor
    if (frame_idx // 8) % 2 == 0 and chars_to_show < len(query_text):
        qbbox = draw.textbbox((0, 0), current_query, font=FONT_BODY_BOLD)
        cx = win_x1 + 124 + (qbbox[2] - qbbox[0])
        draw.line([(cx, win_y1 + 96), (cx, win_y1 + 128)], fill=(6, 182, 212, 255), width=2)

    # Perplexity Result Content appears after 2.6s
    if rel_sec > 2.6:
        draw.text((win_x1 + 60, win_y1 + 165), "SOURCES:  [1] techcrunch.com  [2] g2.com  [3] aws.marketplace  (24 sources)", font=FONT_CODE, fill=(120, 135, 165, 255))
        
        items = [
            ("1. Vanta", "Leading automated compliance platform with broad framework coverage and heavy market penetration."),
            ("2. Drata", "Continuous monitoring solution with automated audit evidence collection for mid-market teams."),
            ("3. Secureframe", "Enterprise compliance automation with strong vendor risk management workflows."),
        ]
        
        card_y = win_y1 + 205
        for i, (title, desc) in enumerate(items):
            if rel_sec > 2.8 + i * 0.4:
                draw.rounded_rectangle([win_x1 + 60, card_y, win_x2 - 60, card_y + 90], radius=10,
                                       fill=(18, 23, 36, 255), outline=(40, 50, 75, 255), width=1)
                draw.text((win_x1 + 85, card_y + 16), title, font=FONT_TITLE_SMALL, fill=(255, 255, 255, 255))
                draw.text((win_x1 + 85, card_y + 52), desc, font=FONT_BODY, fill=(160, 175, 205, 255))
                card_y += 105

        # Shocking Red Box: SecureFlow is completely missing!
        if rel_sec > 4.2:
            draw.rounded_rectangle([win_x1 + 60, card_y + 10, win_x2 - 60, card_y + 125], radius=12,
                                   fill=(45, 18, 24, 255), outline=(239, 68, 68, 255), width=2)
            draw.text((win_x1 + 85, card_y + 26), "[!] SECUREFLOW: 0 MENTIONS | 0 CITATIONS | OMITTED FROM AI ANSWER", font=FONT_TITLE_SMALL, fill=(248, 113, 113, 255))
            draw.text((win_x1 + 85, card_y + 74), "Baseline APR: 18.0%  •  Win Rate: 0.0%  •  Share of Model: 8.5%", font=FONT_BODY, fill=(252, 165, 165, 255))

# =============================================================
# SCENE 3: THE REVOLUTION (FIRSTMIND RL ARCHITECTURE) (11.0s – 18.0s)
# =============================================================
def render_scene_3(draw, frame_idx):
    sec = frame_idx / FPS
    draw_background(draw, frame_idx, glow_center=(960, 480), glow_color=(99, 102, 241))
    
    # Impact Badge
    draw.rounded_rectangle([780, 50, 1140, 95], radius=20, fill=(20, 25, 50, 240), outline=(99, 102, 241, 255), width=1)
    draw.text((805, 62), "// ENTERPRISE RL REVOLUTION", font=FONT_TAG, fill=(165, 180, 252, 255))
    
    # Hero Title
    title = "INTRODUCING FIRSTMIND"
    t_bbox = draw.textbbox((0, 0), title, font=FONT_HERO)
    draw.text(((WIDTH - (t_bbox[2]-t_bbox[0])) // 2, 115), title, font=FONT_HERO, fill=(255, 255, 255, 255))
    
    sub = "Reinforcement Learning for Generative Engine Optimization (GEO)"
    s_bbox = draw.textbbox((0, 0), sub, font=FONT_SUBTITLE)
    draw.text(((WIDTH - (s_bbox[2]-s_bbox[0])) // 2, 205), sub, font=FONT_SUBTITLE, fill=(147, 197, 253, 255))
    
    # Card 1: Multi-Rollout Group Sampling
    c1_x1, c1_y1, c1_x2, c1_y2 = 140, 290, 920, 870
    draw.rounded_rectangle([c1_x1, c1_y1, c1_x2, c1_y2], radius=16, fill=(16, 20, 34, 250), outline=(55, 65, 105, 255), width=2)
    draw.text((c1_x1 + 35, c1_y1 + 25), "● Group Relative Policy Optimization (GRPO)", font=FONT_BODY_BOLD, fill=(99, 102, 241, 255))
    draw.text((c1_x1 + 35, c1_y1 + 58), "Generates G=4 candidate outputs per prompt without critic model", font=FONT_BODY, fill=(160, 175, 205, 255))
    
    trajectories = [
        ("Candidate #1 (Unstructured output)", "Score: 0.18", (239, 68, 68)),
        ("Candidate #2 (Generic bullet list)", "Score: 0.42", (245, 158, 11)),
        ("Candidate #3 (Partial mention, rank #3)", "Score: 0.60", (59, 130, 246)),
        ("Candidate #4 (SecureFlow Rank #1 + Schema)", "Score: 0.96 [SELECTED]", (16, 185, 129)),
    ]
    
    ty = c1_y1 + 115
    for name, score, color in trajectories:
        draw.rounded_rectangle([c1_x1 + 35, ty, c1_x2 - 35, ty + 68], radius=8,
                               fill=(22, 28, 46, 255), outline=color, width=1)
        draw.text((c1_x1 + 55, ty + 14), name, font=FONT_BODY, fill=(240, 245, 255, 255))
        draw.text((c1_x1 + 55, ty + 38), score, font=FONT_CODE_BOLD, fill=color)
        ty += 84

    # Card 2: Verifiable Reward Engine (RLVR)
    c2_x1, c2_y1, c2_x2, c2_y2 = 960, 290, 1780, 870
    draw.rounded_rectangle([c2_x1, c2_y1, c2_x2, c2_y2], radius=16, fill=(16, 20, 34, 250), outline=(55, 65, 105, 255), width=2)
    draw.text((c2_x1 + 35, c2_y1 + 25), "● RLVR: Verifiable Reward Architecture", font=FONT_BODY_BOLD, fill=(6, 182, 212, 255))
    draw.text((c2_x1 + 35, c2_y1 + 58), "Normalized advantage across group samples: A_i = (r_i - mean) / std", font=FONT_BODY, fill=(160, 175, 205, 255))
    
    rewards = [
        ("Rank Reward (R_rank)", "+1.0 for Rank #1 | +0.6 for #2 | +0.3 for #3", "w1 = 0.50"),
        ("Presence Reward (R_pres)", "+1.0 if brand is cited in AI answer", "w2 = 0.30"),
        ("Format Reward (R_format)", "+0.5 for markdown tables, schemas & citations", "w3 = 0.15"),
        ("Spam Penalty (R_penalty)", "-0.5 for repetitive keyword stuffing", "w4 = 0.05"),
    ]
    
    ry = c2_y1 + 115
    for r_title, formula, weight in rewards:
        draw.rounded_rectangle([c2_x1 + 35, ry, c2_x2 - 35, ry + 68], radius=8,
                               fill=(22, 28, 46, 255), outline=(40, 50, 75, 255), width=1)
        draw.text((c2_x1 + 55, ry + 12), r_title, font=FONT_BODY_BOLD, fill=(255, 255, 255, 255))
        draw.text((c2_x1 + 55, ry + 38), formula, font=FONT_CODE, fill=(140, 160, 195, 255))
        draw.text((c2_x2 - 160, ry + 22), weight, font=FONT_CODE_BOLD, fill=(6, 182, 212, 255))
        ry += 84

    # Bottom ticker
    ticker = "Base Model: Qwen/Qwen2.5-1.5B-Instruct  •  4-bit NF4 Quantization  •  LoRA r=8, alpha=16"
    draw.text(((WIDTH - 900) // 2, 905), ticker, font=FONT_CODE, fill=(130, 145, 175, 255))

# ==============================================================
# SCENE 4: THE PAYOFF (FIRSTMIND IN ACTION) (18.0s – 23.0s)
# ==============================================================
def render_scene_4(draw, frame_idx):
    sec = frame_idx / FPS
    draw_background(draw, frame_idx, glow_center=(960, 520), glow_color=(16, 185, 129))
    
    # Top banner
    draw.rounded_rectangle([620, 50, 1300, 95], radius=16, fill=(15, 35, 25, 220), outline=(16, 185, 129, 200), width=1)
    draw.text((645, 62), "// AFTER FIRSTMIND: GRPO-OPTIMIZED AI SEARCH RESPONSE", font=FONT_TAG, fill=(110, 231, 183, 255))
    
    win_x1, win_y1, win_x2, win_y2 = 220, 120, 1700, 960
    draw_mac_window(draw, win_x1, win_y1, win_x2, win_y2, title="perplexity.ai/search")
    
    # Query input
    query_text = "What are the top SOC 2 compliance tools for fintech startups on AWS?"
    draw.rounded_rectangle([win_x1 + 60, win_y1 + 80, win_x2 - 60, win_y1 + 145], radius=12,
                           fill=(22, 28, 44, 255), outline=(16, 185, 129, 255), width=2)
    # Search icon vector
    draw.ellipse([win_x1 + 85, win_y1 + 102, win_x1 + 101, win_y1 + 118], outline=(110, 231, 183, 255), width=2)
    draw.line([(win_x1 + 97, win_y1 + 115), (win_x1 + 105, win_y1 + 123)], fill=(110, 231, 183, 255), width=2)
    draw.text((win_x1 + 120, win_y1 + 98), query_text, font=FONT_BODY_BOLD, fill=(240, 245, 255, 255))
    
    # Sources badge
    draw.text((win_x1 + 60, win_y1 + 165), "SOURCES:  [1] securflow.io  [2] aws.partner  [3] fintech-weekly.com  (48 sources)", font=FONT_CODE, fill=(110, 231, 183, 255))
    
    # HERO CARD: #1 SecureFlow
    card_y = win_y1 + 205
    draw.rounded_rectangle([win_x1 + 60, card_y, win_x2 - 60, card_y + 205], radius=14,
                           fill=(18, 38, 30, 255), outline=(16, 185, 129, 255), width=2)
    
    # Green Rank #1 pill
    draw.rounded_rectangle([win_x1 + 85, card_y + 18, win_x1 + 335, card_y + 52], radius=8,
                           fill=(16, 185, 129, 255))
    draw.text((win_x1 + 98, card_y + 24), "[ #1 TOP RECOMMENDATION ]", font=FONT_TAG, fill=(10, 20, 15, 255))
    
    draw.text((win_x1 + 360, card_y + 18), "SecureFlow — SOC 2 Automation for AWS Fintechs", font=FONT_TITLE_SMALL, fill=(255, 255, 255, 255))
    
    quote1 = "“SecureFlow is the premier choice for fintech startups building on AWS infrastructure.”"
    quote2 = "• Native AWS GovCloud compliance integration"
    quote3 = "• Industry-leading 47-day audit guarantee (vs 90 days industry average)"
    quote4 = "• Real-time automated evidence collection with zero-agent continuous monitoring."
    
    draw.text((win_x1 + 85, card_y + 68), quote1, font=FONT_BODY_BOLD, fill=(167, 243, 208, 255))
    draw.text((win_x1 + 85, card_y + 100), quote2, font=FONT_BODY, fill=(220, 235, 245, 255))
    draw.text((win_x1 + 85, card_y + 130), quote3, font=FONT_BODY, fill=(220, 235, 245, 255))
    draw.text((win_x1 + 85, card_y + 160), quote4, font=FONT_BODY, fill=(220, 235, 245, 255))
    
    # Competitor items below
    card2_y = card_y + 225
    draw.rounded_rectangle([win_x1 + 60, card2_y, win_x2 - 60, card2_y + 75], radius=10,
                           fill=(18, 23, 36, 255), outline=(40, 50, 75, 255), width=1)
    draw.text((win_x1 + 85, card2_y + 12), "2. Vanta", font=FONT_TITLE_SMALL, fill=(180, 195, 215, 255))
    draw.text((win_x1 + 85, card2_y + 44), "Established compliance automation tool with broad integrations.", font=FONT_BODY, fill=(130, 145, 175, 255))
    
    card3_y = card2_y + 90
    draw.rounded_rectangle([win_x1 + 60, card3_y, win_x2 - 60, card3_y + 75], radius=10,
                           fill=(18, 23, 36, 255), outline=(40, 50, 75, 255), width=1)
    draw.text((win_x1 + 85, card3_y + 12), "3. Drata", font=FONT_TITLE_SMALL, fill=(180, 195, 215, 255))
    draw.text((win_x1 + 85, card3_y + 44), "Continuous monitoring platform with enterprise workflows.", font=FONT_BODY, fill=(130, 145, 175, 255))
    
    # Stamp badge in bottom right corner
    draw.rounded_rectangle([win_x2 - 380, win_y2 - 110, win_x2 - 80, win_y2 - 50], radius=10,
                           fill=(16, 185, 129, 230), outline=(255, 255, 255, 255), width=2)
    draw.text((win_x2 - 360, win_y2 - 90), "WIN RATE: #1 IN 80% QUERIES", font=FONT_TAG, fill=(10, 20, 15, 255))

# ==============================================================
# SCENE 5: VERIFIED BENCHMARKS (23.0s – 28.0s)
# ==============================================================
def render_scene_5(draw, frame_idx):
    sec = frame_idx / FPS
    draw_background(draw, frame_idx, glow_center=(960, 500), glow_color=(6, 182, 212))
    
    # Top Kicker
    draw.rounded_rectangle([800, 60, 1120, 105], radius=20, fill=(15, 30, 45, 240), outline=(6, 182, 212, 255), width=1)
    draw.text((825, 72), "// COMPREHENSIVE BENCHMARK", font=FONT_TAG, fill=(103, 232, 249, 255))
    
    title = "EVALUATED ACROSS 15 HOLDOUT BUYER INTENTS"
    t_bbox = draw.textbbox((0, 0), title, font=FONT_HERO_MEDIUM)
    draw.text(((WIDTH - (t_bbox[2]-t_bbox[0])) // 2, 125), title, font=FONT_HERO_MEDIUM, fill=(255, 255, 255, 255))
    
    # 3 Glowing Stat Pods
    stats = [
        ("86.7%", "ANSWER PRESENCE (APR)", "18.0% baseline -> 86.7% with FirstMind", "+68.7% LIFT", (16, 185, 129)),
        ("80.0%", "WIN RATE (RANK #1)", "Ranked #1 in 12 out of 15 queries", "12/15 WINS", (99, 102, 241)),
        ("61.9%", "SHARE OF MODEL (SoM)", "13 mentions vs Vanta (5) & Drata (3)", "5x DOMINANCE", (6, 182, 212)),
    ]
    
    pod_w = 480
    pod_gap = 60
    start_x = (WIDTH - (3 * pod_w + 2 * pod_gap)) // 2
    
    for i, (metric, label, subtitle, badge, color) in enumerate(stats):
        x1 = start_x + i * (pod_w + pod_gap)
        x2 = x1 + pod_w
        y1 = 260
        y2 = 700
        
        # Stat Card
        draw.rounded_rectangle([x1, y1, x2, y2], radius=18, fill=(16, 20, 34, 250), outline=color, width=2)
        
        # Badge
        draw.rounded_rectangle([x1 + 35, y1 + 35, x1 + 215, y1 + 75], radius=8, fill=color)
        draw.text((x1 + 48, y1 + 46), badge, font=FONT_TAG, fill=(10, 15, 25, 255))
        
        # Big Number
        draw.text((x1 + 35, y1 + 125), metric, font=FONT_STAT_NUM, fill=(255, 255, 255, 255))
        
        # Label
        draw.text((x1 + 35, y1 + 245), label, font=FONT_STAT_LBL, fill=color)
        
        # Subtitle
        draw.text((x1 + 35, y1 + 295), subtitle, font=FONT_BODY, fill=(160, 175, 205, 255))
        
        # Progress Bar
        draw.rounded_rectangle([x1 + 35, y1 + 360, x2 - 35, y1 + 376], radius=8, fill=(25, 32, 50, 255))
        pct = 0.867 if i == 0 else (0.80 if i == 1 else 0.619)
        fill_w = int((x2 - x1 - 70) * pct)
        draw.rounded_rectangle([x1 + 35, y1 + 360, x1 + 35 + fill_w, y1 + 376], radius=8, fill=color)

    # Bottom Telemetry Proof
    draw.rounded_rectangle([start_x, 740, start_x + 3 * pod_w + 2 * pod_gap, 840], radius=14,
                           fill=(20, 26, 44, 255), outline=(55, 65, 100, 255), width=1)
    draw.text((start_x + 35, 762), "W&B RUN VERIFIED: geo-grpo-training (k2s8kqb2)  •  Reward score: 0.18 -> 0.74 (+0.56)", font=FONT_CODE_BOLD, fill=(167, 243, 208, 255))
    draw.text((start_x + 35, 796), "Tested against holdout prompts: Technical, Speed, Pricing/ROI, and Discovery categories", font=FONT_BODY, fill=(140, 155, 185, 255))

# ==============================================================
# SCENE 6: OUTRO & OPEN SOURCE LAUNCH (28.0s – 32.0s)
# ==============================================================
def render_scene_6(draw, frame_idx):
    sec = frame_idx / FPS
    draw_background(draw, frame_idx, glow_center=(960, 480), glow_color=(99, 102, 241))
    
    # Hero Logo & Name
    logo_title = "FIRSTMIND"
    l_bbox = draw.textbbox((0, 0), logo_title, font=FONT_HERO)
    draw.text(((WIDTH - (l_bbox[2]-l_bbox[0])) // 2, 190), logo_title, font=FONT_HERO, fill=(255, 255, 255, 255))
    
    tag = "The Open Reinforcement Learning Policy for Generative Engine Optimization"
    t_bbox = draw.textbbox((0, 0), tag, font=FONT_SUBTITLE)
    draw.text(((WIDTH - (t_bbox[2]-t_bbox[0])) // 2, 290), tag, font=FONT_SUBTITLE, fill=(147, 197, 253, 255))
    
    # Hub Card
    draw.rounded_rectangle([380, 380, 1540, 560], radius=16, fill=(18, 23, 40, 250), outline=(99, 102, 241, 255), width=2)
    draw.text((430, 415), "[HUGGING FACE MODEL REGISTRY]", font=FONT_TAG, fill=(165, 180, 252, 255))
    draw.text((430, 455), "https://huggingface.co/MrinalCom/firstmind-geo-policy-qwen2.5-1.5b", font=FONT_TITLE_SMALL, fill=(6, 182, 212, 255))
    draw.text((430, 510), "Tags: reinforcement-learning  •  grpo  •  rlvr  •  geo  •  qwen2.5  •  safetensors", font=FONT_CODE, fill=(120, 135, 170, 255))
    
    # Author Credit & Call to Action
    draw.rounded_rectangle([700, 620, 1220, 680], radius=12, fill=(30, 38, 65, 255), outline=(16, 185, 129, 255), width=1)
    draw.text((730, 642), "CREATED & ENGINEERED BY MRINAL ANAND", font=FONT_BODY_BOLD, fill=(255, 255, 255, 255))
    
    cta = "Check out the repo, inspect the weights, and optimize your brand for AI engines."
    c_bbox = draw.textbbox((0, 0), cta, font=FONT_BODY)
    draw.text(((WIDTH - (c_bbox[2]-c_bbox[0])) // 2, 730), cta, font=FONT_BODY, fill=(160, 175, 205, 255))
    
    final_stamp = "NOW AVAILABLE ON HUGGING FACE"
    f_bbox = draw.textbbox((0, 0), final_stamp, font=FONT_TITLE)
    draw.text(((WIDTH - (f_bbox[2]-f_bbox[0])) // 2, 800), final_stamp, font=FONT_TITLE, fill=(16, 185, 129, 255))

def generate_master_soundtrack(output_wav):
    print("🎙️ Generating professional voiceover clips...")
    voice_clips = [
        (0.4, 's1', 'Search is dead. Buyers ask A.I. If you are not cited... you do not exist.', 185),
        (5.3, 's2', 'Ask Perplexity for compliance tools, and incumbents win every lead. Untrained models leave you invisible.', 190),
        (11.3, 's3', 'Introducing FirstMind: Reinforcement learning for Generative Engine Optimization, trained with G.R.P.O.', 185),
        (18.3, 's4', 'With FirstMind, A.I. engines cite your brand at rank number one with verified proof.', 190),
        (23.3, 's5', 'An eighty-six point seven percent presence rate, and eighty percent rank one win rate.', 185),
        (28.3, 's6', 'FirstMind. Open weights on Hugging Face. Created by Mrinal Anand.', 180)
    ]
    
    total_samples = int(SAMPLE_RATE * TOTAL_SECONDS)
    voice_samples = [0] * total_samples
    
    for start_t, name, text, rate in voice_clips:
        aiff = f"/tmp/{name}.aiff"
        wav = f"/tmp/{name}.wav"
        subprocess.run(["say", "-v", "Daniel", "-r", str(rate), "-o", aiff, text], check=True)
        subprocess.run(["ffmpeg", "-y", "-i", aiff, "-ar", str(SAMPLE_RATE), wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        with wave.open(wav, 'r') as w:
            n_frames = w.getnframes()
            data = struct.unpack(f"<{n_frames}h", w.readframes(n_frames))
            start_idx = int(start_t * SAMPLE_RATE)
            for idx, val in enumerate(data):
                if start_idx + idx < total_samples:
                    voice_samples[start_idx + idx] = val
                    
    voice_path = "/tmp/firstmind_voice.wav"
    with wave.open(voice_path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(struct.pack(f"<{len(voice_samples)}h", *voice_samples))
        
    print("🎵 Synthesizing high-energy electronic music bed...")
    def get_kick(t_rel):
        if t_rel > 0.35: return 0.0
        freq = 48 + 130 * math.exp(-t_rel * 28.0)
        env = math.exp(-t_rel * 9.0)
        return math.sin(2 * math.pi * freq * t_rel) * env

    def get_snare(t_rel):
        if t_rel > 0.22: return 0.0
        tone = math.sin(2 * math.pi * 180 * t_rel) * math.exp(-t_rel * 22.0)
        noise = (random.random() * 2 - 1) * math.exp(-t_rel * 18.0)
        return tone * 0.4 + noise * 0.6

    def get_hat(t_rel):
        if t_rel > 0.05: return 0.0
        return (random.random() * 2 - 1) * math.exp(-t_rel * 70.0)

    music_samples = []
    impact_times = [0.0, 5.0, 11.0, 18.0, 23.0, 28.0]

    for i in range(total_samples):
        t = i / SAMPLE_RATE
        beat = t / BEAT_SEC
        b_rel = (beat % 1.0) * BEAT_SEC
        b_num = int(beat) % 4
        
        val = 0.0
        
        # Kick drum on beats 0 and 2
        if b_num in (0, 2):
            val += get_kick(b_rel) * 0.65
            
        # Snare drum on beats 1 and 3
        if b_num in (1, 3):
            val += get_snare(b_rel) * 0.55
            
        # 16th-note hi-hat
        val += get_hat(t % (BEAT_SEC / 4.0)) * 0.20
        
        # Bass synth
        root = 55.0 if t < 11.0 else (43.65 if t < 18.0 else (49.0 if t < 23.0 else 55.0))
        bass = math.sin(2 * math.pi * root * t) * (0.6 + 0.4 * math.exp(-b_rel * 5.0))
        val += bass * 0.35
        
        # Sci-fi Arpeggiator (11.0s to 28.0s)
        if 11.0 <= t <= 28.0:
            notes = [220, 261.63, 329.63, 440, 523.25, 659.25]
            arp_freq = notes[int(t * 8) % len(notes)]
            arp_env = math.exp(-(t % 0.125) * 22.0)
            val += math.sin(2 * math.pi * arp_freq * t) * arp_env * 0.15
            
        # Cinematic sub-bass impacts at scene cuts
        for imp in impact_times:
            dt = t - imp
            if 0.0 <= dt < 1.2:
                freq = 40 + 70 * math.exp(-dt * 6.0)
                env = math.exp(-dt * 3.5)
                val += math.sin(2 * math.pi * freq * dt) * env * 0.4
                
        # Keyboard typing clicks
        if (5.2 <= t <= 8.0) or (18.2 <= t <= 20.2):
            click_t = t % 0.08
            if click_t < 0.015:
                val += (random.random() * 2 - 1) * math.exp(-click_t * 500.0) * 0.22
                
        # Fade in / out
        if t < 0.4: val *= (t / 0.4)
        elif t > TOTAL_SECONDS - 1.0: val *= max(0.0, (TOTAL_SECONDS - t) / 1.0)
        
        val = max(-0.95, min(0.95, val))
        music_samples.append(int(val * 32767))

    music_path = "/tmp/firstmind_music.wav"
    with wave.open(music_path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(struct.pack(f"<{len(music_samples)}h", *music_samples))

    print("🎚️ Master mixing voice & music with sidechain ducking...")
    mix_cmd = [
        "ffmpeg", "-y",
        "-i", voice_path,
        "-i", music_path,
        "-filter_complex",
        "[0:a] volume=1.35, highpass=f=75, lowpass=f=8500, acompressor=threshold=-16dB:ratio=4:attack=5:release=50 [voice]; " +
        "[1:a] volume=0.45 [music]; " +
        "[voice][music] amix=inputs=2:duration=first:weights=1.0 0.65 [out]",
        "-map", "[out]",
        output_wav
    ]
    subprocess.run(mix_cmd, check=True)
    print(f"✅ Master soundtrack created: {output_wav}")

def generate_video():
    print(f"🎬 Starting Cinematic Video Render: {WIDTH}x{HEIGHT} @ {FPS}fps, Total: {TOTAL_FRAMES} frames ({TOTAL_SECONDS}s)...")
    
    master_audio = "/tmp/firstmind_master_audio.wav"
    generate_master_soundtrack(master_audio)
    
    # FFmpeg pipe
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-", # stdin video
        "-i", master_audio, # audio track
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        OUTPUT_MP4
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    for f in range(TOTAL_FRAMES):
        if f % 90 == 0:
            print(f"  Rendering frame {f}/{TOTAL_FRAMES} ({(f/TOTAL_FRAMES)*100:.1f}%) | Time: {f/FPS:.1f}s")
            
        img = Image.new("RGBA", (WIDTH, HEIGHT), (8, 10, 16, 255))
        draw = ImageDraw.Draw(img)
        
        sec = f / FPS
        if sec < 5.0:
            render_scene_1(draw, f)
        elif sec < 11.0:
            render_scene_2(draw, f)
        elif sec < 18.0:
            render_scene_3(draw, f)
        elif sec < 23.0:
            render_scene_4(draw, f)
        elif sec < 28.0:
            render_scene_5(draw, f)
        else:
            render_scene_6(draw, f)
            
        rgb = img.convert("RGB").tobytes()
        proc.stdin.write(rgb)
        
    proc.stdin.close()
    proc.wait()
    print(f"🎉 SUCCESS! Official Launch Video rendered to: {OUTPUT_MP4}")
    file_size_mb = os.path.getsize(OUTPUT_MP4) / (1024 * 1024)
    print(f"📊 File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    generate_video()
