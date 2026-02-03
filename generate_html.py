#!/usr/bin/env python3
"""
Generate optimized HTML presentation using loops for JPEG and TIFF files.
"""
import os
import glob

# Define file data
jpeg_files = [
    {
        'name': 'cathedral.jpg',
        'g_offset': '(5, 2)',
        'r_offset': '(12, 3)',
        'time': '0.5s'
    },
    {
        'name': 'monastery.jpg', 
        'g_offset': '(-3, 2)',
        'r_offset': '(3, 2)',
        'time': '0.5s'
    },
    {
        'name': 'tobolsk.jpg',
        'g_offset': '(3, 3)', 
        'r_offset': '(6, 3)',
        'time': '0.7s'
    }
]

# Main 11 TIF files from assignment
main_tiff_files = [
    {
        'name': 'church.tif',
        'g_offset': '(25, 4)',
        'r_offset': '(58, -4)', 
        'time': '20.1s',
        'note': ''
    },
    {
        'name': 'emir.tif',
        'g_offset': '(49, 24)',
        'r_offset': '(107, 40)', 
        'time': '18.7s',
        'note': 'Used edge-based features'
    },
    {
        'name': 'harvesters.tif',
        'g_offset': '(60, 17)',
        'r_offset': '(124, 14)',
        'time': '11.5s',
        'note': ''
    },
    {
        'name': 'icon.tif',
        'g_offset': '(42, 17)',
        'r_offset': '(90, 23)',
        'time': '17.3s',
        'note': ''
    },
    {
        'name': 'italil.tif',
        'g_offset': '(38, 22)', 
        'r_offset': '(77, 36)',
        'time': '12.6s',
        'note': ''
    },
    {
        'name': 'lastochikino.tif',
        'g_offset': '(-3, -2)',
        'r_offset': '(76, -8)',
        'time': '11.2s', 
        'note': ''
    },
    {
        'name': 'lugano.tif',
        'g_offset': '(41, -17)',
        'r_offset': '(92, -29)',
        'time': '10.8s',
        'note': ''
    },
        {
        'name': 'melons.tif',
        'g_offset': '(80, 10)',
        'r_offset': '(177, 13)',
        'time': '10.4s',
        'note': ''
    },
    {
        'name': 'self_portrait.tif',
        'g_offset': '(78, 29)',
        'r_offset': '(176, 37)', 
        'time': '10.3s',
        'note': ''
    },
    {
        'name': 'siren.tif',
        'g_offset': '(49, -6)',
        'r_offset': '(96, -24)',
        'time': '10.3s',
        'note': ''
    },
    {
        'name': 'three_generations.tif',
        'g_offset': '(54, 12)',
        'r_offset': '(111, 9)',
        'time': '14.1s',
        'note': ''
    }
]

# Additional 3 master-pnp files 
master_pnp_files = [
    {
        'name': 'master-pnp-prok-00000-00086a.tif',
        'g_offset': '(57, 32)',
        'r_offset': '(129, 49)',
        'time': '10.3s',
        'note': 'Library of Congress collection'
    },
    {
        'name': 'master-pnp-prok-00100-00116a.tif', 
        'g_offset': '(75, 9)',
        'r_offset': '(158, 16)',
        'time': '15.4s',
        'note': 'Library of Congress collection'
    },
    {
        'name': 'master-pnp-prok-00100-00187u.tif',
        'g_offset': '(33, -11)',
        'r_offset': '(140, -26)',
        'time': '10.5s',
        'note': 'Library of Congress collection'
    }
]

# Auto-detect additional files if they exist
def detect_files():
    """Auto-detect all processed files in output directory"""
    output_dir = "output"
    if os.path.exists(output_dir):
        jpg_files = glob.glob(os.path.join(output_dir, "*cathedral*")) + \
                   glob.glob(os.path.join(output_dir, "*monastery*")) + \
                   glob.glob(os.path.join(output_dir, "*tobolsk*"))
        
        tif_files = glob.glob(os.path.join(output_dir, "*.jpg"))
        tif_files = [f for f in tif_files if not any(x in f for x in ['cathedral', 'monastery', 'tobolsk'])]
        
        print(f"Detected {len(jpg_files)} JPEG results, {len(tif_files)} TIFF results")
        return jpg_files, tif_files
    return [], []

def generate_image_card(file_info, method_type):
    """Generate HTML for a single image result card"""
    name = file_info['name']
    base_name = os.path.splitext(name)[0]
    
    method_badge = "single-scale" if method_type == "jpeg" else "multi-scale"
    method_text = "Single-Scale NCC" if method_type == "jpeg" else "Pyramid NCC + Edges" if file_info.get('note') else "Pyramid NCC"
    
    note_line = f"<br><strong>Note:</strong> {file_info['note']}" if file_info.get('note') else ""
    
    return f'''            <div class="image-result">
                <span class="method-badge {method_badge}">{method_text}</span>
                <h4>{name}</h4>
                <img src="output/{base_name}_restored.jpg" alt="{base_name.title()} aligned result">
                <div class="offset-table">
                    Green offset: (dy, dx) = {file_info['g_offset']}<br>
                    Red offset: (dy, dx) = {file_info['r_offset']}<br>
                    Processing time: {file_info['time']}{note_line}
                </div>
            </div>'''

def generate_html():
    """Generate complete HTML file using loops"""
    
    # Generate JPEG section
    jpeg_cards = '\n'.join([generate_image_card(file_info, "jpeg") for file_info in jpeg_files])
    
    # Generate TIFF sections - main 11 TIFs and 3 master-pnp files
    main_tiff_cards = '\n'.join([generate_image_card(file_info, "tiff") for file_info in main_tiff_files])
    master_pnp_cards = '\n'.join([generate_image_card(file_info, "tiff") for file_info in master_pnp_files])
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prokudin-Gorskii Image Alignment Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background-color: #f8f9fa;
        }}
        .header {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            margin: 30px 0;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-result {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: #fafafa;
        }}
        .image-result img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .image-result h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .offset-table {{
            background: #f1f3f4;
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .method-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .single-scale {{
            background: #e3f2fd;
            color: #1565c0;
        }}
        .multi-scale {{
            background: #f3e5f5;
            color: #6a1b9a;
        }}
        .algorithm-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .algorithm-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .bells-whistles {{
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .bells-whistles h3 {{
            margin-top: 0;
            color: #d84315;
        }}
        .feature-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .feature-item {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #ff5722;
        }}
        code {{
            background: #f5f5f5;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        .file-stats {{
            background: #e8f5e8;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #4caf50;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Prokudin-Gorskii Image Alignment</h1>
        <p>Automatic restoration of early color photographs using computer vision</p>
    </div>

    <div class="section">
        <h2>📊 Processing Summary</h2>
        <div class="file-stats">
            <h4>📁 Files Processed:</h4>
            <ul>
                <li><strong>{len(jpeg_files)} JPEG files</strong> using single-scale alignment</li>
                <li><strong>{len(main_tiff_files)} main TIFF files</strong> using multi-scale pyramid alignment</li>
                <li><strong>{len(master_pnp_files)} additional master-pnp files</strong> from Library of Congress</li>
                <li><strong>Total: {len(jpeg_files) + len(main_tiff_files) + len(master_pnp_files)} images</strong> successfully processed</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>🎯 Project Overview</h2>
        <p>This project implements automatic alignment of Sergei Mikhailovich Prokudin-Gorskii's glass plate photographs (1907-1915). Each image contains three separate exposures (Blue, Green, Red) that need precise alignment to create stunning color photographs.</p>
        
        <div class="algorithm-section">
            <div class="algorithm-box">
                <h3>🔍 Single-Scale Alignment</h3>
                <p><strong>For {len(jpeg_files)} JPEG files:</strong> {', '.join([f['name'] for f in jpeg_files])}</p>
                <ul>
                    <li>Exhaustive search over [-15, 15] pixel window</li>
                    <li>Sobel edge features for robustness</li>
                    <li>Normalized Cross-Correlation (NCC) scoring</li>
                    <li>Fast processing for small images</li>
                </ul>
            </div>
            <div class="algorithm-box">
                <h3>🔺 Multi-Scale Pyramid</h3>
                <p><strong>For {len(main_tiff_files) + len(master_pnp_files)} TIFF files:</strong> All high-resolution glass plate scans</p>
                <ul>
                    <li>5-level pyramid with 2x downsampling</li>
                    <li>Coarse-to-fine alignment strategy</li>
                    <li>Handles large displacements efficiently</li>
                    <li>±2 pixel refinement at each level</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📊 Single-Scale Alignment Results ({len(jpeg_files)} JPEG Files)</h2>
        <p>Results using exhaustive search alignment on low-resolution images:</p>
        
        <div class="image-grid">
{jpeg_cards}
        </div>
    </div>

    <div class="section">
        <h2>🔺 Multi-Scale Pyramid Results ({len(main_tiff_files)} Main TIFF Files)</h2>
        <p>Results using pyramid alignment on high-resolution glass plate scans:</p>
        
        <div class="image-grid">
{main_tiff_cards}
        </div>
    </div>

    <div class="section">
        <h2>🌟 Library of Congress Collection ({len(master_pnp_files)} Master-PNP Files)</h2>
        <p>Additional images downloaded from the Prokudin-Gorskii collection:</p>
        
        <div class="image-grid">
{master_pnp_cards}
        </div>
    </div>

    <div class="bells-whistles">
        <h3>🔥 Bells & Whistles Implementation</h3>
        <p>Advanced image enhancement features implemented beyond basic alignment:</p>
        
        <div class="feature-list">
            <div class="feature-item">
                <h4>🖼️ Automatic Border Cropping</h4>
                <p>Detects and removes colored glass plate borders using gradient analysis and edge detection.</p>
            </div>
            
            <div class="feature-item">
                <h4>⚖️ Automatic White Balance</h4>
                <p>Gray World assumption: adjusts channel gains so average color becomes neutral gray.</p>
            </div>
            
            <div class="feature-item">
                <h4>🌈 Automatic Contrast Enhancement</h4>
                <p>Percentile-based histogram stretching (2%-98%) for optimal dynamic range per channel.</p>
            </div>
            
            <div class="feature-item">
                <h4>🔍 Edge-Based Features</h4>
                <p>Sobel gradient magnitude features robust to brightness differences between channels.</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>🔧 Technical Implementation</h2>
        
        <h3>Algorithm Pipeline:</h3>
        <ol>
            <li><strong>Load & Preprocess:</strong> Handle 16-bit TIFF and 8-bit JPEG, normalize for processing</li>
            <li><strong>Channel Splitting:</strong> Divide vertical glass plate into B, G, R thirds</li>
            <li><strong>Feature Extraction:</strong> Compute Sobel edge features for robust matching</li>
            <li><strong>Alignment:</strong> Single-scale (JPEGs) or multi-scale pyramid (TIFFs)</li>
            <li><strong>Reconstruction:</strong> Apply computed offsets to original high-bit-depth channels</li>
            <li><strong>Enhancement:</strong> Auto-crop, white balance, contrast adjustment</li>
        </ol>
        
        <h3>Key Parameters:</h3>
        <ul>
            <li><code>search_range=15</code>: Maximum displacement window</li>
            <li><code>pyramid_depth=5</code>: Number of pyramid levels (handles up to 480px displacement)</li>
            <li><code>crop_ratio=0.15</code>: Border cropping to avoid edge artifacts</li>
            <li><code>NCC metric</code>: Normalized Cross-Correlation for scoring</li>
        </ul>
        
        <h3>Performance Summary:</h3>
        <ul>
            <li><strong>JPEG files ({len(jpeg_files)}):</strong> Average {sum([float(f['time'][:-1]) for f in jpeg_files])/len(jpeg_files):.1f}s (single-scale)</li>
            <li><strong>Main TIFF files ({len(main_tiff_files)}):</strong> Average {sum([float(f['time'][:-1]) for f in main_tiff_files])/len(main_tiff_files):.1f}s (pyramid alignment)</li>
            <li><strong>Master-PNP files ({len(master_pnp_files)}):</strong> Average {sum([float(f['time'][:-1]) for f in master_pnp_files])/len(master_pnp_files):.1f}s (pyramid alignment)</li>
            <li><strong>Memory efficient:</strong> Processes images up to 3000x9000 pixels</li>
            <li><strong>Total processing time:</strong> ~{sum([float(f['time'][:-1]) for f in jpeg_files + main_tiff_files + master_pnp_files]):.0f} seconds for all images</li>
        </ul>
    </div>

    <div class="section">
        <h2>🎓 What We Covered</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #4caf50;">
                ✅ <strong>Single-scale alignment</strong><br>
                Results on {len(jpeg_files)} JPEG files with computed offsets
            </div>
            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #4caf50;">
                ✅ <strong>Multi-scale pyramid alignment</strong><br>
                Results on {len(main_tiff_files)} provided TIFF examples
            </div>
            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #4caf50;">
                ✅ <strong>NCC & L2 metrics</strong><br>
                NCC implemented and used by default
            </div>
            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #4caf50;">
                ✅ <strong>Additional collection images</strong><br>
                {len(master_pnp_files)}+ images from LoC Prokudin-Gorskii collection
            </div>
        </div>
    </div>

    <div style="text-align: center; margin: 40px 0; color: #666;">
        <p>Created for CS Computer Vision Assignment - Prokudin-Gorskii Image Alignment</p>
        <p><strong>Source Code: <a href="https://github.com/Mohak327/rgb-img-merging/tree/main">https://github.com/Mohak327/rgb-img-merging/tree/main</a></strong></p>
        <p><strong>Submitted By: Mohak Sharma (ms7306)</strong></p>
    </div>
</body>
</html>'''

    return html_content

if __name__ == "__main__":
    print("Generating optimized HTML with loops...")
    html = generate_html()
    
    with open("index.html", "w", encoding='utf-8') as f:
        f.write(html)
    
    print(f"Generated index.html with {len(jpeg_files)} JPEG, {len(main_tiff_files)} main TIFF, and {len(master_pnp_files)} master-pnp results")
    print("HTML generated using efficient loop-based approach!")
    
    # Show all files being processed
    print("\n📁 All Files Processed:")
    print("JPEG Files (Single-Scale):")
    for i, file_info in enumerate(jpeg_files, 1):
        print(f"  {i}. {file_info['name']} - G{file_info['g_offset']}, R{file_info['r_offset']} - {file_info['time']}")
    
    print(f"\nMain TIFF Files (Multi-Scale Pyramid):")
    for i, file_info in enumerate(main_tiff_files, 1):
        note = f" - {file_info['note']}" if file_info.get('note') else ""
        print(f"  {i}. {file_info['name']} - G{file_info['g_offset']}, R{file_info['r_offset']} - {file_info['time']}{note}")
    
    print(f"\nMaster-PNP Files (Library of Congress):")
    for i, file_info in enumerate(master_pnp_files, 1):
        note = f" - {file_info['note']}" if file_info.get('note') else ""
        print(f"  {i}. {file_info['name']} - G{file_info['g_offset']}, R{file_info['r_offset']} - {file_info['time']}{note}")
    
    print(f"\n✅ Total: {len(jpeg_files) + len(main_tiff_files) + len(master_pnp_files)} images processed with optimized loop structure!")