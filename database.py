from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

DATABASE_URL = "sqlite:///./vehicles.db"  # change to postgres later

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    pool_pre_ping=True  # Auto-reconnect on connection loss
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class VehicleLog(Base):
    __tablename__ = "vehicle_logs"  # Fixed: was **tablename**
    
    id = Column(Integer, primary_key=True, index=True)
    vehicle_number = Column(String(20), index=True, nullable=False)
    driver_name = Column(String(100), nullable=True)
    vehicle_type = Column(String(50), nullable=True)
    entry_time = Column(DateTime, default=datetime.now, nullable=False)
    exit_time = Column(DateTime, default=None, nullable=True)
    status = Column(String(10), default="ENTRY")  # ENTRY or EXIT
    operator_id = Column(String(50), default="system")
    image_path = Column(Text, nullable=True)
    gate_id = Column(String(50), default="main_gate")
    remarks = Column(Text, default="")
    
    def __repr__(self):
        return f"<VehicleLog(id={self.id}, vehicle={self.vehicle_number}, status={self.status})>"

# Create tables
Base.metadata.create_all(bind=engine)