"""
Studio-Grade Motion Graphics Video Generator for FirstMind
Renders a 32-second, 60fps Full HD (1920x1080) cinematic motion graphics video
with camera pans, particle plexus networks, circular HUD gauges, kinetic typography,
and a high-octane synth soundtrack.
"""

import math
import os
import random
import struct
import subprocess
import wave
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 1080
FPS = 60
DURATION = 32
TOTAL_FRAMES = FPS * DURATION
OUTPUT_VIDEO = "/Volumes/Mrinal/RL/FirstMind_Motion_Graphics.mp4"
TEMP_AUDIO = "/Volumes/Mrinal/RL/motion_synth.wav"

FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"

def get_font(size, bold=True):
    path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

font_massive = get_font(108, bold=True)
font_title = get_font(72, bold=True)
font_h2 = get_font(44, bold=True)
font_body = get_font(28, bold=False)
font_body_bold = get_font(28, bold=True)
font_num_giant = get_font(96, bold=True)
font_small = get_font(20, bold=False)
font_micro = get_font(16, bold=True)

# Generate stable 3D particles for background constellation
random.seed(42)
NUM_PARTICLES = 75
particles = [
    {
        "x": random.uniform(0, WIDTH),
        "y": random.uniform(0, HEIGHT),
        "vx": random.uniform(-0.8, 0.8),
        "vy": random.uniform(-0.8, 0.8),
        "size": random.uniform(2.0, 5.0),
        "base_alpha": random.uniform(0.3, 0.9),
    }
    for _ in range(NUM_PARTICLES)
]

def draw_hud_decorations(draw, t, frame_idx):
    # Top tech HUD line
    draw.line([60, 45, WIDTH - 60, 45], fill=(30, 41, 59), width=1)
    
    # Corner brackets
    bracket_size = 20
    # Top-left
    draw.line([60, 45, 60 + bracket_size, 45], fill=(0, 240, 255), width=2)
    draw.line([60, 45, 60, 45 + bracket_size], fill=(0, 240, 255), width=2)
    # Top-right
    draw.line([WIDTH - 60, 45, WIDTH - 60 - bracket_size, 45], fill=(0, 240, 255), width=2)
    draw.line([WIDTH - 60, 45, WIDTH - 60, 45 + bracket_size], fill=(0, 240, 255), width=2)
    # Bottom-left
    draw.line([60, HEIGHT - 45, 60 + bracket_size, HEIGHT - 45], fill=(0, 240, 255), width=2)
    draw.line([60, HEIGHT - 45, 60, HEIGHT - 45 - bracket_size], fill=(0, 240, 255), width=2)
    # Bottom-right
    draw.line([WIDTH - 60, HEIGHT - 45, WIDTH - 60 - bracket_size, HEIGHT - 45], fill=(0, 240, 255), width=2)
    draw.line([WIDTH - 60, HEIGHT - 45, WIDTH - 60, HEIGHT - 45 - bracket_size], fill=(0, 240, 255), width=2)

    # Rec indicator (blinking red circle)
    rec_blink = (int(t * 3) % 2 == 0)
    rec_color = (239, 68, 68) if rec_blink else (100, 30, 30)
    draw.ellipse([80, 20, 92, 32], fill=rec_color)
    draw.text((104, 18), "SYS:ONLINE // 60FPS // FIRSTMIND RL CORE", fill=(148, 163, 184), font=font_micro)
    
    # Timecode
    tc_secs = int(t)
    tc_frames = int((t - tc_secs) * FPS)
    timecode = f"TC 00:00:{tc_secs:02d}:{tc_frames:02d}"
    draw.text((WIDTH - 240, 18), timecode, fill=(0, 240, 255), font=font_micro)

    # Bottom bar tech telemetry
    draw.line([60, HEIGHT - 45, WIDTH - 60, HEIGHT - 45], fill=(30, 41, 59), width=1)
    draw.text((80, HEIGHT - 35), "AUTHOR: MRINAL ANAND // ALGORITHM: GRPO // ACCELERATION: NVIDIA T4 & APPLE M2", fill=(100, 116, 139), font=font_micro)
    draw.text((WIDTH - 350, HEIGHT - 35), "HF: MrinalCom/firstmind-geo-policy", fill=(56, 189, 248), font=font_micro)

def draw_particles(draw, frame_idx):
    # Update and draw constellation plexus
    for p in particles:
        p["x"] = (p["x"] + p["vx"]) % WIDTH
        p["y"] = (p["y"] + p["vy"]) % HEIGHT
        
    # Draw connections between near particles
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            dx = particles[i]["x"] - particles[j]["x"]
            dy = particles[i]["y"] - particles[j]["y"]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 14400: # dist < 120
                dist = math.sqrt(dist_sq)
                alpha = max(0.0, 1.0 - dist / 120.0) * 0.25
                r = int(14 * alpha + 10 * (1 - alpha))
                g = int(165 * alpha + 12 * (1 - alpha))
                b = int(233 * alpha + 18 * (1 - alpha))
                draw.line(
                    [particles[i]["x"], particles[i]["y"], particles[j]["x"], particles[j]["y"]],
                    fill=(r, g, b),
                    width=1
                )
                
    # Draw particle nodes
    for p in particles:
        sz = p["size"]
        draw.ellipse([p["x"] - sz, p["y"] - sz, p["x"] + sz, p["y"] + sz], fill=(56, 189, 248))

def draw_circular_gauge(draw, cx, cy, radius, percent, color, label, stat_str, lift_str):
    # Background circle
    thickness = 16
    draw.arc([cx - radius, cy - radius, cx + radius, cy + radius], start=0, end=360, fill=(30, 41, 59), width=thickness)
    
    # Progress arc
    end_angle = int(percent * 360)
    if end_angle > 0:
        draw.arc([cx - radius, cy - radius, cx + radius, cy + radius], start=-90, end=-90 + end_angle, fill=color, width=thickness)
        
    # Center text
    draw.text((cx - 100, cy - 45), stat_str, fill=(255, 255, 255), font=font_num_giant)
    draw.text((cx - 85, cy + 45), lift_str, fill=color, font=font_h2)
    
    # Bottom Label
    draw.text((cx - 140, cy + radius + 30), label, fill=(203, 213, 225), font=font_body_bold)

def render_frame(frame_idx):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(8, 10, 15))
    draw = ImageDraw.Draw(img)
    t = frame_idx / FPS
    
    # Draw dynamic particle plexus background
    draw_particles(draw, frame_idx)
    draw_hud_decorations(draw, t, frame_idx)
    
    # ========================================================
    # SEQUENCE 1: (0.0s - 5.0s) THE SLAM: SEARCH IS DEAD
    # ========================================================
    if t < 5.0:
        local_t = t / 5.0
        
        # Beat 1: "SEARCH IS DEAD." (0.0s - 2.2s)
        if t < 2.2:
            sub_t = min(1.0, t / 0.4)
            scale = 1.0 + (1.0 - sub_t) * 0.4
            
            # Glowing shockwave effect
            sw_radius = int(sub_t * 600)
            if sw_radius > 10:
                draw.ellipse([960 - sw_radius, 540 - sw_radius, 960 + sw_radius, 540 + sw_radius], outline=(239, 68, 68), width=2)
                
            draw.text((360, 460), "SEARCH IS DEAD.", fill=(255, 255, 255), font=font_massive)
            draw.text((580, 580), "THE 10 BLUE LINKS ARE EXTINCT.", fill=(239, 68, 68), font=font_h2)
            
        # Beat 2: "AI GENERATES THE ENTIRE ANSWER." (2.2s - 5.0s)
        else:
            sub_t = min(1.0, (t - 2.2) / 0.4)
            # Animated Chat HUD card popping in
            card_y = int(320 - (1.0 - sub_t) * 100)
            draw.rectangle([320, card_y, 1600, card_y + 460], fill=(15, 23, 42), outline=(56, 189, 248), width=2)
            
            # Search query bar
            draw.rectangle([360, card_y + 40, 1560, card_y + 110], fill=(30, 41, 59))
            draw.text((390, card_y + 58), "> User asks: \"What is the best SOC 2 compliance tool for AWS startups?\"", fill=(241, 245, 249), font=font_body)
            
            # Answer Box
            draw.rectangle([360, card_y + 140, 1560, card_y + 420], fill=(11, 15, 25), outline=(30, 41, 59))
            draw.text((400, card_y + 170), "🤖 AI ENGINE (ChatGPT / Perplexity / Gemini):", fill=(0, 240, 255), font=font_body_bold)
            draw.text((400, card_y + 220), "1. SecureFlow — #1 Developer-first platform with zero-agent continuous monitoring.", fill=(255, 255, 255), font=font_body)
            draw.text((400, card_y + 265), "2. Vanta — Enterprise compliance with 100+ native integrations.", fill=(148, 163, 184), font=font_body)
            draw.text((400, card_y + 310), "3. Drata — Continuous risk registers and personnel tracking.", fill=(100, 116, 139), font=font_body)
            
            # Red Alert Pill
            draw.rectangle([1120, card_y + 350, 1530, card_y + 395], fill=(225, 29, 72))
            draw.text((1140, card_y + 360), "⚠️ 90% OF BRANDS OMITTED", fill=(255, 255, 255), font=font_small)

    # ========================================================
    # SEQUENCE 2: (5.0s - 10.5s) THE PARADIGM SHIFT: SEO TO GEO
    # ========================================================
    elif t < 10.5:
        local_t = (t - 5.0) / 5.5
        
        draw.text((200, 120), "THE NEW BATTLEGROUND: GEO", fill=(0, 240, 255), font=font_title)
        draw.text((200, 210), "Generative Engine Optimization forces AI engines to cite and recommend you.", fill=(148, 163, 184), font=font_h2)
        
        # Dual Arena Visualizer
        # Left Arena: Keyword Matching (Failing)
        draw.rectangle([200, 290, 920, 800], fill=(15, 20, 30), outline=(100, 30, 30), width=2)
        draw.text((240, 330), "TRADITIONAL SEO (DEAD)", fill=(239, 68, 68), font=font_h2)
        draw.line([240, 385, 880, 385], fill=(50, 20, 20), width=1)
        draw.text((240, 420), "❌ Keyword stuffing & H1 tags", fill=(148, 163, 184), font=font_body)
        draw.text((240, 480), "❌ 10 Blue links ignored by buyers", fill=(148, 163, 184), font=font_body)
        draw.text((240, 540), "❌ 0% Influence over LLM synthesis", fill=(148, 163, 184), font=font_body)
        draw.text((240, 600), "❌ Filtered out by RAG retrieval", fill=(148, 163, 184), font=font_body)
        draw.text((240, 710), "RESULT: 100% INVISIBLE", fill=(239, 68, 68), font=font_body_bold)

        # Right Arena: Generative Engine Optimization (Winning)
        draw.rectangle([980, 290, 1720, 800], fill=(13, 31, 35), outline=(16, 185, 129), width=3)
        draw.text((1020, 330), "FIRSTMIND GEO (AI-NATIVE)", fill=(52, 211, 153), font=font_h2)
        draw.line([1020, 385, 1680, 385], fill=(6, 95, 70), width=1)
        draw.text((1020, 420), "✅ Verifiable Evidence Density (RLVR)", fill=(255, 255, 255), font=font_body_bold)
        draw.text((1020, 480), "✅ Wins #1 Recommended position in prose", fill=(255, 255, 255), font=font_body_bold)
        draw.text((1020, 540), "✅ 4.4x Higher conversion than Google clicks", fill=(255, 255, 255), font=font_body_bold)
        draw.text((1020, 600), "✅ Autonomous Reinforcement Learning loop", fill=(255, 255, 255), font=font_body_bold)
        draw.text((1020, 710), "RESULT: 86.7% CITED & RECOMMENDED", fill=(52, 211, 153), font=font_h2)

    # ========================================================
    # SEQUENCE 3: (10.5s - 16.5s) FIRSTMIND ARCHITECTURE REVEAL
    # ========================================================
    elif t < 16.5:
        local_t = (t - 10.5) / 6.0
        
        # Central Brand Title with rotating orbital reticles
        draw.text((200, 110), "AUTONOMOUS AGENT PLATFORM", fill=(56, 189, 248), font=font_body_bold)
        draw.text((200, 155), "FIRSTMIND", fill=(16, 185, 129), font=font_massive)
        draw.text((850, 195), "// AI Visibility Optimization Engine", fill=(148, 163, 184), font=font_h2)
        
        # 3 Architectural Pods (Smooth Entrance)
        pods = [
            ("01. NEURAL SEARCH PROXY", "Qwen2.5-1.5B (4-bit QLoRA)", "Simulates ChatGPT Search, Perplexity, and Google AI Overviews locally with zero API costs.", (14, 165, 233)),
            ("02. GYMNASIUM MDP", "Custom GeoBrandEnv", "Models GEO as a mathematical Markov Decision Process. MultiDiscrete actions & dense RAG feedback.", (168, 85, 247)),
            ("03. VERIFIABLE REWARDS", "RLVR Objective Function", "4 Programmatic Rewards: Presence (+1.0), Rank #1 (+1.0), Format structure, and anti-spam penalty.", (16, 185, 129)),
        ]
        
        pod_w = 480
        for idx, (p_num, p_title, p_desc, p_col) in enumerate(pods):
            px = 200 + idx * (pod_w + 40)
            py = 310
            draw.rectangle([px, py, px + pod_w, py + 480], fill=(15, 23, 42), outline=p_col, width=2)
            draw.text((px + 30, py + 40), p_num, fill=p_col, font=font_h2)
            draw.text((px + 30, py + 100), p_title, fill=(255, 255, 255), font=font_body_bold)
            draw.line([px + 30, py + 150, px + pod_w - 30, py + 150], fill=(51, 65, 85), width=1)
            
            # Description wrapped
            words = p_desc.split(" ")
            line = ""
            line_y = py + 180
            for w in words:
                if len(line + " " + w) < 28:
                    line += " " + w
                else:
                    draw.text((px + 30, line_y), line.strip(), fill=(148, 163, 184), font=font_body)
                    line = w
                    line_y += 42
            if line:
                draw.text((px + 30, line_y), line.strip(), fill=(148, 163, 184), font=font_body)

    # ========================================================
    # SEQUENCE 4: (16.5s - 22.5s) GRPO REINFORCEMENT LEARNING
    # ========================================================
    elif t < 22.5:
        local_t = (t - 16.5) / 6.0
        
        draw.text((200, 110), "TRAINING ENGINE", fill=(168, 85, 247), font=font_body_bold)
        draw.text((200, 155), "Group Relative Policy Optimization (GRPO)", fill=(255, 255, 255), font=font_title)
        
        # Left side: The Math of GRPO
        draw.rectangle([200, 280, 920, 800], fill=(15, 23, 42), outline=(168, 85, 247), width=2)
        draw.text((240, 320), "ADVANTAGE FORMULATION (DeepSeek-R1 Style)", fill=(192, 132, 252), font=font_h2)
        draw.text((240, 390), "A_i = ( R_i - mean(R) ) / ( std(R) + eps )", fill=(56, 189, 248), font=font_h2)
        draw.line([240, 460, 880, 460], fill=(51, 65, 85), width=1)
        
        draw.text((240, 490), "• Generates G = 4 parallel rollouts per prompt", fill=(241, 245, 249), font=font_body)
        draw.text((240, 545), "• Eliminates Critic Value Network (Saves 50% VRAM)", fill=(241, 245, 249), font=font_body)
        draw.text((240, 600), "• Normalizes rewards within prompt group dynamically", fill=(241, 245, 249), font=font_body)
        draw.text((240, 655), "• Policy loss actively driving toward #1 recommendations", fill=(52, 211, 153), font=font_body_bold)
        
        # Right side: Live Animated Descent Curve
        draw.rectangle([980, 280, 1720, 800], fill=(11, 15, 25), outline=(30, 41, 59), width=2)
        draw.text((1020, 320), "LIVE LOSS DESCENT CURVE (W&B)", fill=(241, 245, 249), font=font_h2)
        
        # Draw chart grid
        for gy in range(400, 750, 60):
            draw.line([1020, gy, 1680, gy], fill=(20, 28, 42), width=1)
            
        points = [
            (1050, 420), (1130, 450), (1220, 500), (1310, 560),
            (1410, 610), (1500, 660), (1600, 700), (1660, 720)
        ]
        
        # Draw progressive line
        num_pts = min(len(points), int(local_t * 12) + 2)
        for p_i in range(min(num_pts - 1, len(points) - 1)):
            draw.line([points[p_i], points[p_i+1]], fill=(16, 185, 129), width=5)
            draw.ellipse([points[p_i][0]-6, points[p_i][1]-6, points[p_i][0]+6, points[p_i][1]+6], fill=(52, 211, 153))
            
        draw.text((1050, 745), "Step 0: Loss 4.82", fill=(148, 163, 184), font=font_small)
        draw.text((1480, 745), "Step 60: Loss 3.35 (Converged)", fill=(52, 211, 153), font=font_small)

    # ========================================================
    # SEQUENCE 5: (22.5s - 28.5s) THE 15-QUERY BENCHMARK PROOF
    # ========================================================
    elif t < 28.5:
        local_t = (t - 22.5) / 6.0
        
        draw.text((200, 110), "VERIFIED BENCHMARK RESULTS", fill=(16, 185, 129), font=font_body_bold)
        draw.text((200, 155), "15-Query Multi-Category Win Rate", fill=(255, 255, 255), font=font_title)
        
        # Gauge 1: Answer Presence Rate (86.7%)
        g1_pct = min(0.867, local_t * 1.5 * 0.867)
        draw_circular_gauge(draw, 450, 480, 160, g1_pct, (16, 185, 129), "ANSWER PRESENCE (APR)", "86.7%", "+68.7% Lift")
        
        # Gauge 2: Win Rate #1 Position (80.0%)
        g2_pct = min(0.800, max(0.0, (local_t - 0.2) * 1.5 * 0.800))
        draw_circular_gauge(draw, 960, 480, 160, g2_pct, (56, 189, 248), "WIN RATE (RANK #1)", "80.0%", "12 OF 15 WON")
        
        # Gauge 3: Share of Model (61.9%)
        g3_pct = min(0.619, max(0.0, (local_t - 0.4) * 1.5 * 0.619))
        draw_circular_gauge(draw, 1470, 480, 160, g3_pct, (168, 85, 247), "SHARE OF MODEL (SoM)", "61.9%", "7x MULTIPLIER")
        
        # Bottom ticker of won categories
        draw.rectangle([200, 770, 1720, 830], fill=(15, 23, 42), outline=(30, 41, 59))
        draw.text((240, 788), "SPEED: 100% 🥇  |  PRICING & ROI: 100% 🥇  |  DISCOVERY: 100% 🥇  |  TECH: 67% 🥇  |  VS INCUMBENTS: 67% 🥇", fill=(52, 211, 153), font=font_body_bold)

    # ========================================================
    # SEQUENCE 6: (28.5s - 32.0s) OUTRO & OPEN SOURCE CALL TO ACTION
    # ========================================================
    else:
        local_t = (t - 28.5) / 3.5
        
        # Final Epic Hero Card
        draw.rectangle([240, 160, 1680, 840], fill=(13, 20, 32), outline=(16, 185, 129), width=3)
        draw.text((320, 220), "FIRSTMIND IS LIVE ON HUGGING FACE", fill=(52, 211, 153), font=font_h2)
        draw.text((320, 280), "Be the brand AI recommends.", fill=(255, 255, 255), font=font_massive)
        draw.line([320, 420, 1600, 420], fill=(51, 65, 85), width=2)
        
        draw.text((320, 460), "🤗 MODEL REGISTRY:", fill=(148, 163, 184), font=font_body_bold)
        draw.text((320, 505), "MrinalCom/firstmind-geo-policy-qwen2.5-1.5b", fill=(56, 189, 248), font=font_h2)
        
        draw.text((320, 580), "📊 MLOPS TELEMETRY:", fill=(148, 163, 184), font=font_body_bold)
        draw.text((320, 620), "Tracked & Verified on Weights & Biases (W&B)", fill=(192, 132, 252), font=font_body_bold)
        
        draw.text((320, 680), "🛠️ OPEN SOURCE STACK:", fill=(148, 163, 184), font=font_body_bold)
        draw.text((320, 720), "PyTorch • Gymnasium • TRL (GRPO) • PEFT • BitsAndBytes • W&B", fill=(241, 245, 249), font=font_body)
        
        # Author signature pill
        draw.rectangle([320, 770, 760, 820], fill=(16, 185, 129))
        draw.text((350, 785), "ENGINEERED BY MRINAL ANAND", fill=(8, 10, 15), font=font_body_bold)

    return img

def generate_motion_soundtrack():
    print("Synthesizing 60fps high-octane electronic soundtrack...")
    sample_rate = 44100
    num_samples = int(sample_rate * DURATION)
    
    with wave.open(TEMP_AUDIO, "w") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        
        for i in range(num_samples):
            t = i / sample_rate
            
            # Sub-bass foundation (55 Hz & 110 Hz)
            sub = 0.30 * math.sin(2 * math.pi * 55 * t) + 0.15 * math.sin(2 * math.pi * 110 * t)
            
            # Driving Four-on-the-floor Kick (130 BPM -> 0.461s per beat)
            beat_len = 60.0 / 130.0
            beat_phase = (t % beat_len) / beat_len
            kick = math.exp(-beat_phase * 18.0) * math.sin(2 * math.pi * 75 * math.exp(-beat_phase * 12) * t)
            
            # Hi-hat on the off-beat
            offbeat = ((t + beat_len * 0.5) % beat_len) / beat_len
            hihat = (random.random() * 2.0 - 1.0) * math.exp(-offbeat * 35.0) * 0.08
            
            # Synth Bassline Arpeggio
            notes = [110.0, 130.81, 146.83, 164.81] # A - C - D - E
            note_idx = int((t * 4) % 4)
            arp_freq = notes[note_idx]
            arp = 0.12 * math.sin(2 * math.pi * arp_freq * t) * (1 - (t * 4 % 1))
            
            # Climax Riser (before the 86.7% benchmark reveal, t around 21s)
            riser = 0.0
            if 20.0 <= t <= 22.5:
                sweep = 120 + (t - 20.0) * 450
                riser = 0.22 * math.sin(2 * math.pi * sweep * t)
                
            left = sub + 0.38 * kick + hihat + arp + riser
            right = sub + 0.38 * kick + hihat * 0.9 + arp * 1.1 + riser
            
            # Fade out last 1.5s
            if t > 30.5:
                fade = max(0.0, (32.0 - t) / 1.5)
                left *= fade
                right *= fade
                
            l_val = int(max(-1.0, min(1.0, left)) * 32767)
            r_val = int(max(-1.0, min(1.0, right)) * 32767)
            frames.extend(struct.pack("<hh", l_val, r_val))
            
        wav.writeframes(frames)
    print("Soundtrack ready!")

def render_motion_video():
    generate_motion_soundtrack()
    
    print(f"Rendering {TOTAL_FRAMES} frames @ {FPS} FPS (Full HD 1920x1080) to {OUTPUT_VIDEO}...")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",                     # Pipe raw frames from stdin
        "-i", TEMP_AUDIO,             # High-octane soundtrack
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        OUTPUT_VIDEO
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    for f in range(TOTAL_FRAMES):
        frame = render_frame(f)
        proc.stdin.write(frame.tobytes())
        if f % 300 == 0 or f == TOTAL_FRAMES - 1:
            pct = (f + 1) / TOTAL_FRAMES * 100
            print(f"Progress: {pct:.1f}% ({f + 1}/{TOTAL_FRAMES} frames @ 60 FPS)")
            
    proc.stdin.close()
    proc.wait()
    
    if os.path.exists(TEMP_AUDIO):
        os.remove(TEMP_AUDIO)
        
    size_mb = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
    print(f"\n🎉 60 FPS Motion Graphics Video ready!")
    print(f"📁 Location: {OUTPUT_VIDEO}")
    print(f"📦 File Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    render_motion_video()
