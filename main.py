from resolver.resolverr import HospitalDomainResolver

if __name__ == "__main__":
    resolver = HospitalDomainResolver()

    result = resolver.resolve(
        hospital_name="St. Vincent's East",
        state="AL"
    )

    print(result)
