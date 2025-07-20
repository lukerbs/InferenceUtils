from inferenceutils import HardwareInspector, Recommender

# Run hardware inspection
inspector = HardwareInspector()
hardware_data = inspector.inspect_all()

# Get engine recommendation
recommender = Recommender()
recommendation = recommender.recommend(hardware_data)

# Add recommendation to data
hardware_data["recommended_engine"] = recommendation

print(f"Recommended engine: {recommendation['name']}")
print(f"Reason: {recommendation['reason']}")
