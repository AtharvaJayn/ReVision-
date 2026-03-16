import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
import database
from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient
from io import BytesIO
import random

load_dotenv()

# --- Configuration ---
st.set_page_config(page_title="ReVision", page_icon="🌱", layout="wide")

GO_API_KEY = os.getenv("GO_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

database.init_db()
genai.configure(api_key=GO_API_KEY)
hf_client = InferenceClient(token=HF_TOKEN)

MODEL_PATH = 'eco_sorter.h5'
LABELS = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# --- MULTIPLE UPCYCLING PROMPTS FOR VARIETY ---
UPCYCLING_IDEAS = {
    'Plastic': [
        "Upcycled plastic bottle vertical garden planter with vibrant flowers, hanging wall mount, natural daylight, photorealistic, eco-friendly DIY",
        "Creative plastic bottle bird feeder with colorful paint, wooden perch, garden setting, detailed craftsmanship, bright photography",
        "Artistic plastic bottle chandelier with LED lights, bohemian decor style, warm ambient lighting, creative home decor",
        "DIY plastic bottle pencil holder with geometric patterns, colorful desk accessories, student workspace, clean background"
    ],
    'Glass': [
        "Elegant upcycled glass bottle terrarium with succulents, miniature garden, natural wood base, soft window lighting, detailed macro photography",
        "Vintage glass bottle vase with wildflowers, rustic farmhouse aesthetic, wooden table, natural sunlight, artistic composition",
        "Repurposed wine bottle candle holder with dripping wax, romantic dinner setting, dark moody lighting, professional photography",
        "Upcycled glass bottle lamp with Edison bulb, industrial chic decor, exposed brick background, warm amber glow"
    ],
    'Metal': [
        "Artistic metal can desk organizer painted in pastel colors, modern office setup, minimalist design, bright natural light",
        "Repurposed tin can hanging planters with herbs, vertical kitchen garden, rustic farmhouse style, natural lighting",
        "Creative aluminum can wind chimes with beads, outdoor garden decor, gentle breeze, soft focus photography",
        "DIY metal can lantern with cutout patterns, outdoor evening ambiance, warm candle glow, cozy backyard setting"
    ],
    'Paper': [
        "Handmade recycled paper notebooks with botanical prints, artisan craft, natural textures, soft studio lighting",
        "Origami paper flower bouquet in ceramic vase, colorful vibrant petals, artistic arrangement, clean white background",
        "Upcycled paper mache bowl with geometric patterns, home decor accent, modern interior, professional product photography",
        "Recycled paper gift boxes with ribbon, eco-friendly packaging, minimalist aesthetic, soft natural light"
    ],
    'Cardboard': [
        "Cardboard box cat house with windows and door, playful pet furniture, cozy interior, natural lighting, cute photography",
        "Repurposed cardboard organizer with compartments, tidy desk setup, minimalist workspace, clean professional photo",
        "Upcycled cardboard wall shelf with plants, boho decor style, natural wood accents, warm home interior"
    ]
}

# --- FUNCTIONS ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def process_image(img):
    img = img.resize((224, 224)) 
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_eco_advice(waste_type):
    if waste_type == "Trash":
        prompt = "My AI classified this as Trash. Give me a check to see if it's actually recyclable, and if not, how to reduce this waste."
    else:
        prompt = f"""
        I have waste classified as: {waste_type}.
        Provide markdown response:
        ### 1. ♻️ How to Recycle (3 strict rules)
        ### 2. 💡 Creative Upcycling Idea (1 DIY project)
        """
    model_gen = genai.GenerativeModel('gemini-flash-latest')
    response = model_gen.generate_content(prompt)
    return response.text

def get_random_upcycling_prompt(waste_type):
    """Get a random upcycling prompt from the list"""
    ideas = UPCYCLING_IDEAS.get(waste_type, [
        f"Creative upcycled {waste_type.lower()} craft project, eco-friendly DIY, professional photo"
    ])
    return random.choice(ideas)

def generate_upcycling_image(waste_type):
    """Generate a unique upcycled image - NO CACHING"""
    try:
        prompt = get_random_upcycling_prompt(waste_type)
        seed = random.randint(1, 1000000)
        
        image = hf_client.text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell",
        )
        return image, prompt
        
    except Exception as e:
        st.error(f"❌ Image generation failed: {str(e)}")
        return None, None

# --- LOAD MODEL ---
try:
    model = load_model()
except Exception as e:
    st.error(f"Model not found. Error: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("👤 User Profile")
    
    if 'username' not in st.session_state:
        st.session_state.username = "Guest"
    
    user_input = st.text_input("Enter Username:", value=st.session_state.username)
    
    if user_input and user_input != st.session_state.username:
        st.session_state.username = user_input
        database.add_user(user_input)
        st.toast(f"Welcome, {user_input}!", icon="👋")

    current_score, total_scans = database.get_user_stats(st.session_state.username)
    
    c1, c2 = st.columns(2)
    with c1: 
        # Create an empty placeholder for the score
        score_placeholder = st.empty() 
        score_placeholder.metric("Score", current_score)
    with c2: 
        # Create an empty placeholder for the scans
        scans_placeholder = st.empty() 
        scans_placeholder.metric("Scans", total_scans)
        
    st.markdown("---")
    st.markdown("### 🏆 Leaderboard")
    
    # Placeholder for the leaderboard so it can update too!
    leaderboard_placeholder = st.empty()
    
    def render_leaderboard():
        top_players = database.get_leaderboard()
        board_text = ""
        for rank, (name, score) in enumerate(top_players):
            board_text += f"**#{rank+1} {name}**: {score} pts\n\n"
        leaderboard_placeholder.markdown(board_text)
        
    render_leaderboard()

# --- MAIN UI ---
st.title("🌱 ReVision: From Waste to Wonder")

uploaded_file = st.file_uploader("Upload an image of waste...", type=["jpg", "png", "jpeg"])

# --- CONDITIONAL HERO SECTION ---
if uploaded_file is None:
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background-color: #f0fdf4; border-radius: 15px; margin-bottom: 2rem; border: 1px solid #dcfce3;">
        <h1 style="font-size: 3.5rem; font-weight: 900; color: #166534; line-height: 1.1; margin-bottom: 1rem;">
            Waste is a Design Flaw.<br><span style="color: #15803d;">Let's Fix It.</span>
        </h1>
        <p style="font-size: 1.25rem; color: #334155; max-width: 800px; margin: 0 auto; line-height: 1.6; padding: 0 1rem;">
            Every minute, <strong>3,825 tons</strong> of municipal waste are generated globally. 
            By 2050, that number will hit an astounding <strong>3.8 billion tons</strong>. 
            Currently, only 9% of plastic successfully cycles back into our economy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error("### 🌊 2,000 Trucks\nThe equivalent of 2,000 garbage trucks full of plastic leak into our oceans every single *day*.")
    with col2:
        st.warning("### ♻️ Only 9%\nGlobally, only 9% of plastic waste is successfully recycled, while 22% is completely mismanaged.")
    with col3:
        st.success("### 🎯 1 Solution\nOur vision AI sorts your waste instantly and provides highly specific, localized upcycling blueprints.")

    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #475569;'>Snap a photo. <b>Sort it right.</b> Build something new.</h3>", unsafe_allow_html=True)

# --- ANALYSIS UI ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Item', use_container_width=True)
    
    if st.button("🔍 Analyze Waste"):
        with st.spinner("Identifying material..."):
            processed_img = process_image(image)
            predictions = model.predict(processed_img)
            predicted_index = np.argmax(predictions)
            predicted_class = LABELS[predicted_index]
            confidence = predictions[0][predicted_index]
        
        if predicted_class == "Trash":
            st.error(f"**Detected: {predicted_class}** ({confidence*100:.1f}%)")
            st.markdown("### 🗑️ Action Required")
            st.warning("⚠️ **Please throw this item in the Dustbin.**")
            st.info("This item is not suitable for recycling or upcycling.")
            
            if st.session_state.username != "Guest":
                database.update_score(st.session_state.username, 2)
                st.toast("✅ +2 Points for responsible disposal!", icon="🚮")
                
                # UPDATE UI DYNAMICALLY
                new_score, new_scans = database.get_user_stats(st.session_state.username)
                score_placeholder.metric("Score", new_score)
                scans_placeholder.metric("Scans", new_scans)
                render_leaderboard()

        else:
            st.success(f"**Detected Material:** {predicted_class} ({confidence*100:.1f}%)")
            
            if st.session_state.username != "Guest":
                database.update_score(st.session_state.username, 10)
                st.toast("⭐ +10 Points!", icon="⭐")
                
                # UPDATE UI DYNAMICALLY
                new_score, new_scans = database.get_user_stats(st.session_state.username)
                score_placeholder.metric("Score", new_score)
                scans_placeholder.metric("Scans", new_scans)
                render_leaderboard()

            # AI Advice
            with st.spinner("Generating eco-advice..."):
                try:
                    advice = get_eco_advice(predicted_class)
                    st.markdown("---")
                    st.markdown(advice)
                except Exception as e:
                    st.error(f"AI Error: {e}")
            
            # Generate Upcycling Image
            st.markdown("---")
            st.markdown("### 🎨 Upcycling Inspiration")
            
            with st.spinner("🖼️ Generating creative upcycling idea..."):
                generated_img, used_prompt = generate_upcycling_image(predicted_class)
                
                if generated_img:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📸 Your Original Item")
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### ✨ Upcycling Inspiration")
                        st.image(generated_img, use_container_width=True)
                        with st.expander("🎯 See prompt used"):
                            st.caption(used_prompt)
                    
                    buf = BytesIO()
                    generated_img.save(buf, format="PNG")
                    st.download_button(
                        label="💾 Download Upcycling Idea",
                        data=buf.getvalue(),
                        file_name=f"upcycled_{predicted_class.lower()}_{random.randint(1,999)}.png",
                        mime="image/png"
                    )
                    
                    if st.button("🔄 Generate Another Idea"):
                        st.rerun()
                    
                    st.success("💡 Transform your waste into something wonderful!")