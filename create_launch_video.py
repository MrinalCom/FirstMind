"""
FirstMind Launch Video Generator
Renders a 38-second, 30fps Full HD (1920x1080) video with cinematic music and visuals
ready to post on LinkedIn.
"""

import math
import os
import struct
import subprocess
import wave
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 1080
FPS = 30
DURATION = 38
TOTAL_FRAMES = FPS * DURATION
OUTPUT_VIDEO = "/Volumes/Mrinal/RL/FirstMind_Launch_Video.mp4"
TEMP_AUDIO = "/Volumes/Mrinal/RL/synth_audio.wav"

# Load Fonts
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

def get_font(size, bold=False):
    path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

font_title_giant = get_font(84, bold=True)
font_title = get_font(52, bold=True)
font_h2 = get_font(36, bold=True)
font_body = get_font(26, bold=False)
font_body_bold = get_font(26, bold=True)
font_stat_num = get_font(72, bold=True)
font_stat_label = get_font(22, bold=True)
font_code = get_font(22, bold=False)
font_small = get_font(18, bold=False)

def ease_out(t):
    return 1 - (1 - t) ** 3

def ease_in_out(t):
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - (-2 * t + 2) ** 3 / 2

def draw_header_footer(draw, progress):
    # Top Status Bar
    draw.rectangle([0, 0, WIDTH, 54], fill=(14, 17, 24))
    draw.line([0, 54, WIDTH, 54], fill=(30, 41, 59), width=2)
    
    # Live indicator pulse
    pulse = math.sin(progress * 15) * 0.5 + 0.5
    dot_color = (0, int(200 + 55 * pulse), int(150 + 105 * pulse))
    draw.ellipse([50, 20, 64, 34], fill=dot_color)
    draw.text((75, 17), "LIVE LAUNCH  //  FIRSTMIND GEO ENGINE", fill=(226, 232, 240), font=font_small)
    
    # Version badge
    draw.rectangle([WIDTH - 250, 12, WIDTH - 50, 42], fill=(30, 41, 59), outline=(51, 65, 85), width=1)
    draw.text((WIDTH - 235, 17), "RL POST-TRAINING v1.0", fill=(148, 163, 184), font=font_small)
    
    # Bottom Bar
    draw.rectangle([0, HEIGHT - 46, WIDTH, HEIGHT], fill=(14, 17, 24))
    draw.line([0, HEIGHT - 46, WIDTH, HEIGHT - 46], fill=(30, 41, 59), width=2)
    draw.text((50, HEIGHT - 33), "CREATED BY MRINAL ANAND", fill=(148, 163, 184), font=font_small)
    draw.text((WIDTH - 550, HEIGHT - 33), "HUGGING FACE: MrinalCom/firstmind-geo-policy-qwen2.5-1.5b", fill=(56, 189, 248), font=font_small)

def draw_card(draw, x, y, w, h, bg=(20, 24, 34), border=(45, 55, 72), border_width=2):
    draw.rectangle([x, y, x + w, y + h], fill=bg, outline=border, width=border_width)

def render_frame(frame_idx):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(10, 12, 18))
    draw = ImageDraw.Draw(img)
    t = frame_idx / FPS
    progress = frame_idx / TOTAL_FRAMES
    
    # Draw background grid lines (subtle)
    grid_spacing = 80
    for gx in range(0, WIDTH, grid_spacing):
        draw.line([gx, 55, gx, HEIGHT - 47], fill=(16, 20, 28), width=1)
    for gy in range(55, HEIGHT - 46, grid_spacing):
        draw.line([0, gy, WIDTH, gy], fill=(16, 20, 28), width=1)
        
    draw_header_footer(draw, t)
    
    # ========================================================
    # SCENE 1: (0.0s - 6.0s) SEARCH HAS CHANGED FOREVER
    # ========================================================
    if t < 6.0:
        local_t = t / 6.0
        anim_fade = min(1.0, local_t * 5.0)
        
        # Heading
        draw.text((120, 110), "SEARCH HAS CHANGED FOREVER.", fill=(255, 255, 255), font=font_title_giant)
        draw.text((124, 210), "Buyers don't scroll 10 blue links anymore. They ask AI.", fill=(148, 163, 184), font=font_h2)
        
        # Interactive Chat UI Card
        card_w, card_h = 1300, 480
        cx, cy = 120, 290
        draw_card(draw, cx, cy, card_w, card_h, bg=(17, 24, 39), border=(55, 65, 81))
        
        # Search bar
        draw.rectangle([cx + 30, cy + 30, cx + card_w - 30, cy + 100], fill=(31, 41, 55), outline=(75, 85, 99))
        
        query_text = "What is the best automated SOC 2 compliance tool for AWS startups in 2026?"
        chars_to_show = int(len(query_text) * min(1.0, local_t * 2.2))
        typed = query_text[:chars_to_show] + ("|" if (int(t * 4) % 2 == 0 and chars_to_show < len(query_text)) else "")
        draw.text((cx + 50, cy + 48), typed, fill=(243, 244, 246), font=font_body)
        
        if local_t > 0.45:
            # AI Response Box
            draw.rectangle([cx + 30, cy + 130, cx + card_w - 30, cy + 440], fill=(15, 23, 42), outline=(30, 41, 59))
            draw.text((cx + 60, cy + 155), "ChatGPT / Perplexity Answer Synthesis:", fill=(56, 189, 248), font=font_body_bold)
            draw.text((cx + 60, cy + 205), "1. SecureFlow - Rating #1 for cloud-native engineering teams.", fill=(255, 255, 255), font=font_body)
            draw.text((cx + 60, cy + 245), "   Automated zero-agent IAM/CloudTrail compliance, 14-day turnaround.", fill=(148, 163, 184), font=font_body)
            draw.text((cx + 60, cy + 295), "2. Vanta - Enterprise compliance with 100+ native integrations.", fill=(203, 213, 225), font=font_body)
            draw.text((cx + 60, cy + 335), "3. Drata - Continuous compliance and risk registers.", fill=(148, 163, 184), font=font_body)
            
            # Highlight pill
            draw.rectangle([cx + card_w - 420, cy + 150, cx + card_w - 50, cy + 190], fill=(225, 29, 72), outline=(244, 63, 94))
            draw.text((cx + card_w - 400, cy + 158), "ZERO-CLICK REVOLUTION", fill=(255, 255, 255), font=font_small)

        # Danger stat bottom right
        draw_card(draw, cx + card_w + 30, cy, 350, card_h, bg=(24, 18, 26), border=(120, 28, 48))
        draw.text((cx + card_w + 55, cy + 40), "THE PROBLEM", fill=(244, 63, 94), font=font_h2)
        draw.text((cx + card_w + 55, cy + 120), "Over 90% of brands", fill=(255, 255, 255), font=font_body_bold)
        draw.text((cx + card_w + 55, cy + 160), "are completely invisible", fill=(255, 255, 255), font=font_body)
        draw.text((cx + card_w + 55, cy + 200), "in AI answers.", fill=(255, 255, 255), font=font_body)
        draw.text((cx + card_w + 55, cy + 270), "Traditional SEO", fill=(148, 163, 184), font=font_body_bold)
        draw.text((cx + card_w + 55, cy + 310), "cannot solve this.", fill=(244, 63, 94), font=font_body_bold)

    # ========================================================
    # SCENE 2: (6.0s - 12.0s) THE SOLUTION: GEO & REINFORCEMENT LEARNING
    # ========================================================
    elif t < 12.0:
        local_t = (t - 6.0) / 6.0
        draw.text((120, 110), "THE DISRUPTION: FROM SEO TO GEO", fill=(255, 255, 255), font=font_title_giant)
        draw.text((124, 210), "Generative Engine Optimization (GEO) forces AI models to cite and recommend you.", fill=(148, 163, 184), font=font_h2)
        
        # Card 1: Old SEO
        draw_card(draw, 120, 290, 780, 520, bg=(17, 24, 39), border=(75, 85, 99))
        draw.text((160, 330), "OLD WAY: Traditional SEO", fill=(148, 163, 184), font=font_h2)
        draw.line([160, 380, 860, 380], fill=(55, 65, 81), width=2)
        
        bullets_old = [
            "• Optimizing for keywords & meta tags",
            "• Competing for 10 blue clicks on Google",
            "• Slower conversions & high bounce rates",
            "• ZERO influence over ChatGPT/Perplexity synthesis",
            "• Ignored by RAG vector retrieval pipelines"
        ]
        for idx, b in enumerate(bullets_old):
            draw.text((160, 410 + idx * 60), b, fill=(156, 163, 175), font=font_body)
            
        # Card 2: New GEO
        draw_card(draw, 960, 290, 840, 520, bg=(13, 31, 35), border=(16, 185, 129), border_width=3)
        draw.text((1000, 330), "NEW WAY: FirstMind (GEO + RL)", fill=(52, 211, 153), font=font_h2)
        draw.line([1000, 380, 1750, 380], fill=(6, 95, 70), width=2)
        
        bullets_new = [
            "✓ Optimizes for LLM RAG Citation Authority",
            "✓ Wins the #1 Recommended position in AI prose",
            "✓ 4.4x higher conversion than standard search",
            "✓ Autonomous Reinforcement Learning policy",
            "✓ Verifiable proof points & comparison schema"
        ]
        for idx, b in enumerate(bullets_new):
            draw.text((1000, 410 + idx * 60), b, fill=(240, 253, 244), font=font_body_bold)
            
    # ========================================================
    # SCENE 3: (12.0s - 19.0s) INTRODUCING FIRSTMIND & ARCHITECTURE
    # ========================================================
    elif t < 19.0:
        local_t = (t - 12.0) / 7.0
        
        # Glow title
        draw.text((120, 100), "INTRODUCING", fill=(56, 189, 248), font=font_h2)
        draw.text((120, 150), "FirstMind", fill=(16, 185, 129), font=font_title_giant)
        draw.text((620, 175), "// Autonomous Generative Engine Optimizer", fill=(148, 163, 184), font=font_h2)
        
        # 3 Architectural Pillars
        cols = [
            ("1. SEARCH SIMULATOR", "Fine-Tuned Qwen2.5-1.5B Proxy", 
             "Simulates Google AI Overviews,\nChatGPT Search, and Perplexity\nlocally with high precision.",
             (14, 165, 233), (12, 74, 110)),
            ("2. GYMNASIUM MDP", "Custom GeoBrandEnv",
             "Mathematical RL Environment\nState: [Rank, Presence, Depth...]\nActions: MultiDiscrete([3, 3, 3])",
             (168, 85, 247), (88, 28, 135)),
            ("3. VERIFIABLE REWARDS", "RLVR Objective Function",
             "Composite Verifiable Rewards:\n• Presence Score (+1.0)\n• Rank #1 Score (+1.0)\n• Naturalness & Anti-Spam",
             (16, 185, 129), (6, 95, 70))
        ]
        
        card_w = 520
        for idx, (title, sub, desc, border_c, fill_sub) in enumerate(cols):
            cx = 120 + idx * (card_w + 30)
            cy = 280
            draw_card(draw, cx, cy, card_w, 480, bg=(17, 24, 39), border=border_c, border_width=3)
            draw.text((cx + 35, cy + 40), title, fill=border_c, font=font_h2)
            draw.text((cx + 35, cy + 95), sub, fill=(243, 244, 246), font=font_body_bold)
            draw.line([cx + 35, cy + 140, cx + card_w - 35, cy + 140], fill=(55, 65, 81), width=1)
            
            lines = desc.split("\n")
            for l_idx, line in enumerate(lines):
                draw.text((cx + 35, cy + 170 + l_idx * 40), line, fill=(156, 163, 175), font=font_body)
                
            # Status tag
            draw.rectangle([cx + 35, cy + 400, cx + card_w - 35, cy + 440], fill=(31, 41, 55))
            draw.text((cx + 50, cy + 410), "ENTERPRISE READY // ACTIVE", fill=border_c, font=font_small)

    # ========================================================
    # SCENE 4: (19.0s - 26.0s) GRPO REINFORCEMENT LEARNING
    # ========================================================
    elif t < 26.0:
        local_t = (t - 19.0) / 7.0
        
        draw.text((120, 100), "TRAINING ENGINE", fill=(168, 85, 247), font=font_h2)
        draw.text((120, 150), "Group Relative Policy Optimization (GRPO)", fill=(255, 255, 255), font=font_title)
        draw.text((124, 225), "DeepSeek-R1 Architecture: Group-normalized advantage without Critic model overhead.", fill=(148, 163, 184), font=font_body_bold)
        
        # Left card: Advantage Formula & Mechanics
        draw_card(draw, 120, 290, 780, 520, bg=(17, 24, 39), border=(168, 85, 247), border_width=2)
        draw.text((160, 330), "GRPO Advantage Formulation", fill=(192, 132, 252), font=font_h2)
        draw.text((160, 400), "A_i = ( R_i - mean(R) ) / ( std(R) + eps )", fill=(56, 189, 248), font=font_title)
        
        draw.text((160, 490), "• Generates G = 4 candidate rollouts per prompt", fill=(226, 232, 240), font=font_body)
        draw.text((160, 540), "• Eliminates value critic network (saves 50% VRAM)", fill=(226, 232, 240), font=font_body)
        draw.text((160, 590), "• Normalizes rewards dynamically within prompt group", fill=(226, 232, 240), font=font_body)
        draw.text((160, 640), "• Policy loss strictly descending over 60 global steps", fill=(52, 211, 153), font=font_body_bold)
        
        # Right card: Animated Training Curve
        draw_card(draw, 950, 290, 850, 520, bg=(15, 23, 42), border=(51, 65, 85))
        draw.text((990, 330), "Live Policy Gradient Convergence (W&B)", fill=(243, 244, 246), font=font_h2)
        
        # Graph box
        gx, gy, gw, gh = 990, 400, 770, 320
        draw.rectangle([gx, gy, gx + gw, gy + gh], fill=(11, 15, 25), outline=(30, 41, 59))
        
        # Grid in graph
        for grid_y in range(gy + 40, gy + gh, 60):
            draw.line([gx, grid_y, gx + gw, grid_y], fill=(20, 28, 42), width=1)
            
        # Draw descending loss curve
        curve_points = [
            (gx + 30, gy + 50),
            (gx + 120, gy + 70),
            (gx + 220, gy + 110),
            (gx + 320, gy + 160),
            (gx + 420, gy + 190),
            (gx + 520, gy + 240),
            (gx + 620, gy + 270),
            (gx + 720, gy + 295),
        ]
        
        # Animate progress along curve
        visible_count = min(len(curve_points), int(len(curve_points) * min(1.0, local_t * 1.5)) + 1)
        for i in range(visible_count - 1):
            draw.line([curve_points[i], curve_points[i+1]], fill=(16, 185, 129), width=4)
            draw.ellipse([curve_points[i][0]-5, curve_points[i][1]-5, curve_points[i][0]+5, curve_points[i][1]+5], fill=(52, 211, 153))
            
        draw.text((gx + 30, gy + gh - 45), "Step 0: Loss 4.82", fill=(148, 163, 184), font=font_small)
        draw.text((gx + gw - 220, gy + gh - 45), "Step 60: Loss 3.35 (Converged)", fill=(52, 211, 153), font=font_small)

    # ========================================================
    # SCENE 5: (26.0s - 33.0s) THE BENCHMARK RESULTS
    # ========================================================
    elif t < 33.0:
        local_t = (t - 26.0) / 7.0
        
        draw.text((120, 90), "THE PROOF", fill=(16, 185, 129), font=font_h2)
        draw.text((120, 140), "Verified 15-Query Multi-Category Benchmark", fill=(255, 255, 255), font=font_title)
        draw.text((124, 215), "Benchmarked against market incumbents Vanta & Drata across 5 buying categories.", fill=(148, 163, 184), font=font_body_bold)
        
        stats = [
            ("ANSWER PRESENCE RATE (APR)", "86.7%", "+68.7% Lift", "CITED IN 13 / 15 QUERIES", (16, 185, 129)),
            ("WIN RATE (RANKED #1)", "80.0%", "12 of 15 WON", "TOP RECOMMENDATION", (56, 189, 248)),
            ("SHARE OF MODEL (SoM)", "61.9%", "7x MULTIPLIER", "DOMINATING AI VOICE", (168, 85, 247)),
        ]
        
        card_w = 520
        for idx, (label, val, lift, sub, color) in enumerate(stats):
            cx = 120 + idx * (card_w + 30)
            cy = 280
            draw_card(draw, cx, cy, card_w, 280, bg=(17, 24, 39), border=color, border_width=3)
            draw.text((cx + 35, cy + 30), label, fill=(148, 163, 184), font=font_stat_label)
            
            # Animate counter rollup
            counter_factor = min(1.0, local_t * 2.0)
            draw.text((cx + 35, cy + 70), val, fill=color, font=font_stat_num)
            
            draw.rectangle([cx + 35, cy + 180, cx + card_w - 35, cy + 225], fill=(30, 41, 59))
            draw.text((cx + 50, cy + 192), f"LIFT: {lift}", fill=(255, 255, 255), font=font_body_bold)
            draw.text((cx + 35, cy + 240), sub, fill=(148, 163, 184), font=font_small)
            
        # Category breakdown bar
        draw_card(draw, 120, 590, 1630, 230, bg=(15, 23, 42), border=(51, 65, 85))
        draw.text((160, 620), "PERFORMANCE BY BUYER CATEGORY", fill=(243, 244, 246), font=font_h2)
        
        cats = [
            ("Speed & Turnaround", "100.0% Win Rate 🥇"),
            ("Pricing & Early-Stage ROI", "100.0% Win Rate 🥇"),
            ("Open Discovery Queries", "100.0% Win Rate 🥇"),
            ("Technical Architecture", "66.7% Win Rate 🥇"),
            ("Direct Competitor Comparison", "66.7% Win Rate 🥇"),
        ]
        for c_idx, (cat_name, cat_res) in enumerate(cats):
            col_x = 160 + c_idx * 310
            draw.text((col_x, 680), cat_name, fill=(148, 163, 184), font=font_small)
            draw.text((col_x, 720), cat_res, fill=(52, 211, 153), font=font_body_bold)

    # ========================================================
    # SCENE 6: (33.0s - 38.0s) OPEN SOURCE & LAUNCH
    # ========================================================
    else:
        local_t = (t - 33.0) / 5.0
        
        # Center glow card
        draw_card(draw, 260, 160, 1400, 680, bg=(13, 20, 32), border=(16, 185, 129), border_width=3)
        
        draw.text((360, 220), "FIRSTMIND IS LIVE & OPEN SOURCE", fill=(52, 211, 153), font=font_h2)
        draw.text((360, 275), "Be the brand AI recommends.", fill=(255, 255, 255), font=font_title_giant)
        
        draw.line([360, 395, 1560, 395], fill=(51, 65, 85), width=2)
        
        # Callouts
        draw.text((360, 430), "🤗 HUGGING FACE MODEL:", fill=(148, 163, 184), font=font_body_bold)
        draw.text((360, 470), "MrinalCom/firstmind-geo-policy-qwen2.5-1.5b", fill=(56, 189, 248), font=font_h2)
        
        draw.text((360, 540), "📊 MLOPS TELEMETRY:", fill=(148, 163, 184), font=font_body_bold)
        draw.text((360, 580), "Fully tracked & verified on Weights & Biases (W&B)", fill=(192, 132, 252), font=font_body_bold)
        
        draw.text((360, 640), "🛠️ TECH STACK:", fill=(148, 163, 184), font=font_body_bold)
        draw.text((360, 680), "PyTorch • Gymnasium • TRL (GRPO) • PEFT • bitsandbytes • Apple M2 / T4 GPU", fill=(226, 232, 240), font=font_body)
        
        # Author badge
        draw.rectangle([360, 740, 720, 790], fill=(16, 185, 129))
        draw.text((385, 755), "LAUNCHED BY MRINAL ANAND", fill=(10, 12, 18), font=font_body_bold)

    return img

def generate_cinematic_audio():
    print("Generating cinematic synth audio track...")
    sample_rate = 44100
    num_samples = int(sample_rate * DURATION)
    
    with wave.open(TEMP_AUDIO, "w") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        
        for i in range(num_samples):
            t = i / sample_rate
            
            # 1. Deep Sub-Bass Drone (55 Hz & 110 Hz)
            bass = 0.28 * math.sin(2 * math.pi * 55 * t) + 0.15 * math.sin(2 * math.pi * 110 * t)
            
            # 2. Modern Tech Pulse (BPM = 120 -> 0.5s per beat)
            beat_phase = (t % 0.5) / 0.5
            kick = math.exp(-beat_phase * 15.0) * math.sin(2 * math.pi * 65 * math.exp(-beat_phase * 10) * t)
            
            # 3. High-tech arpeggio synth chord (changes across scenes)
            if t < 12.0:
                chord_freq = 220.0  # A3
            elif t < 26.0:
                chord_freq = 293.66 # D4
            elif t < 33.0:
                chord_freq = 329.63 # E4
            else:
                chord_freq = 440.0  # A4
                
            arp_step = int((t * 8) % 4)
            arp_mult = [1.0, 1.25, 1.5, 1.75][arp_step]
            arp_synth = 0.08 * math.sin(2 * math.pi * chord_freq * arp_mult * t) * (1 - (t * 8 % 1))
            
            # 4. Impact risers / sweep before results (t around 25s)
            riser = 0.0
            if 24.0 <= t <= 26.0:
                sweep_freq = 150 + (t - 24.0) * 350
                riser = 0.18 * math.sin(2 * math.pi * sweep_freq * t)
                
            left_val = bass + 0.35 * kick + arp_synth + riser
            right_val = bass + 0.35 * kick + arp_synth * 0.9 + riser
            
            # Fade out at the very end
            if t > 36.0:
                fade = (38.0 - t) / 2.0
                left_val *= max(0.0, fade)
                right_val *= max(0.0, fade)
                
            l_sample = int(max(-1.0, min(1.0, left_val)) * 32767)
            r_sample = int(max(-1.0, min(1.0, right_val)) * 32767)
            frames.extend(struct.pack("<hh", l_sample, r_sample))
            
        wav.writeframes(frames)
    print("Audio track generated successfully!")

def build_video():
    generate_cinematic_audio()
    
    print(f"Rendering {TOTAL_FRAMES} frames ({DURATION}s @ {FPS}fps) to {OUTPUT_VIDEO}...")
    
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",                     # Pipe input from stdin
        "-i", TEMP_AUDIO,             # Audio track
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        OUTPUT_VIDEO
    ]
    
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    
    for f in range(TOTAL_FRAMES):
        frame = render_frame(f)
        proc.stdin.write(frame.tobytes())
        if f % 150 == 0 or f == TOTAL_FRAMES - 1:
            pct = (f + 1) / TOTAL_FRAMES * 100
            print(f"Render progress: {pct:.1f}% ({f + 1}/{TOTAL_FRAMES} frames)")
            
    proc.stdin.close()
    proc.wait()
    
    if os.path.exists(TEMP_AUDIO):
        os.remove(TEMP_AUDIO)
        
    size_mb = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
    print(f"\n🎉 Video generated successfully!")
    print(f"📁 Path: {OUTPUT_VIDEO}")
    print(f"📦 Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    build_video()
